---
title: 'vLLM 源码追踪（五）· KV cache 上：一条请求的 KV block 是怎么拿到手的'
title_en: 'Tracing vLLM Source (5) · KV Cache I: How a Request Gets Its KV Blocks'
date: 2026-09-09
permalink: /posts/2026/09/vllm-kv-cache-1/
tags:
  - LLM
  - vLLM
  - 源码追踪
---

<div class="lang lang-zh" markdown="1">

本文基于 vLLM v0.21.0（V1 引擎，离线 `LLM.generate` 路径，关闭 multiprocessing，prefix caching 默认开启），用 debugger 追踪 `KVCacheManager.allocate_slots()`：scheduler 说"这一步算 N 个 token"，谁把 N 变成"要几块、拿哪几块、哪几块登记进 prefix cache"。文中行号以该版本为准。

前三篇在 scheduler 里看"这一步算多少"。scheduler 每算出一个 `num_new_tokens`，都要问一次 KV cache manager：块够不够？够就拿，不够就抢占别人或者排队。这一篇看被问的 KV cache manager。

配置改回真实块数：`max_model_len=2048`、`gpu_memory_utilization=0.1`，不再限 10 块。三次 generate 和前面一样：gen1 冷启动 / gen2 同一条 prompt / gen3 两条并发。

调试器里几个数，先摆出来：

- block pool 一共 **33758** 块，可用 **33757**
- gen1 从 waiting 进到 running 要 **2** 块，拿到 `[1, 2]`，同一次调用里 block 1 的 hash 就登记进了 prefix cache，此时模型一步都没跑
- gen2 命中了 block 1，却仍然报"要 2 块"，free 少了 2，可 `new_blocks` 只有 `[7]`
- gen2 跑完，cache 从 5 条变成 **6** 条，同一条 prompt、temperature 0

## 位置：三层，一个函数

```
scheduler.schedule()
├── :702  waiting 段  allocate_slots(request, num_new_tokens, num_new_computed_tokens=…, new_computed_blocks=…, …)
└── :425  running 段  allocate_slots(request, num_new_tokens, num_lookahead_tokens=…)
        └── KVCacheManager.allocate_slots()          kv_cache_manager.py:225
              └── KVCacheCoordinator                 kv_cache_coordinator.py   每个 kv_cache_group 转发一次
                    └── SingleTypeKVCacheManager     single_type_kv_cache_manager.py   真正的块数减法
                          └── BlockPool              block_pool.py   free 队列 + cached_block_hash_to_block
```

我选用的 Qwen2.5-0.5B 全是 full attention，只有一个 kv_cache_group，所以 coordinator 是 `UnitaryKVCacheCoordinator`，下面一个 `FullAttentionManager`。coordinator 这一层今天可以当透明的，它就是 `for manager in self.single_type_managers` 转发。

两个入口传的参数不一样。waiting 段那次带着 `new_computed_blocks`（prefix cache 命中了哪些块），running 段那次只有"这步算几个"。KV cache manager 第一件事就是把它们合成两个数：

```python
num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
total_computed_tokens = min(num_local_computed_tokens + num_external_computed_tokens, self.max_model_len)
```

docstring 里那张 layout 图（:262-294）值得看一眼，它把一条请求的 token 序列切成五段：

```
Blocks layout:
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                          |            < to be allocated >           |
----------------------------------------------------------------------
                          | < to be cached (roughly, |
                          | details below)>          |
----------------------------------------------------------------------
| Prefix-cached tokens from either vLLM   |
| or connector. Can be safely removed if  |
| they are outside sliding window.        |
----------------------------------------------------------------------
|   < cached by vLLM >    | not cached by |
                          | vLLM, but     |
| ref_cnt  | ref_cnt not  | cached by     |
| increased| increased yet| connector     |
----------------------------------------------------------------------
```

- **comp**：之前 step 已经算过的，块早就有了。
- **new_comp**：这次刚从 prefix cache 命中的，块在 cache 里，要挂到这条请求名下。
- **ext_comp**：KV connector 从外部拿的，我们没开，一律 0。
- **new**：这次要算的 token，块要新拿。
- **lookahead**：spec decode 的 draft 槽位，我们 0。

图下面三行说的是这个函数的三个动作对应哪几段：

- "to be computed" = new + lookahead
- "to be allocated" = new_comp + ext_comp + new + lookahead，也就是 comp 右边的全部
- "to be cached" ≈ 到 new 为止，不含 lookahead，因为 draft 可能被拒

**block 1 的 hash 在模型算之前就登记进 prefix cache 了。** 我们现在还在 `schedule()` 里，第一个 forward 都没跑。hash 只依赖 token id 和前一块的 hash，hash 在 Request 建好和追加 token 时就算好了，存在 `request.block_hashes`，不依赖 KV 值。KV 值是这一步 forward 写进去的，等别的请求命中它时，forward 早已跑完。

gen1 进门：comp 0、new_comp 0、new 25。gen2 进门：comp 0、new_comp 16、new 9。整篇文章就是这两组数怎么变成块。

## 第一段：要几块

```python
# kv_cache_manager.py:351-353
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, self.max_model_len)
```

gen1 进门 `num_tokens_need_slot = 0 + 25 = 25`。接着问 coordinator 要几块，真正的算式在 single_type_kv_cache_manager：

```python
num_required_blocks = cdiv(num_tokens, self.block_size)        # cdiv(25, 16) = 2
num_req_blocks = len(self.req_to_blocks.get(request_id, ()))   # 0，名下还没块

if request_id in self.num_cached_block:                        # running 请求走这条快路径
    return max(num_required_blocks - num_req_blocks, 0)

num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks   # 命中块 + 已有块
num_new_blocks = max(num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0)
num_evictable_blocks = self._get_num_evictable_blocks(new_computed_blocks[...])
return num_new_blocks + num_evictable_blocks
```

一句话：**要几块 = cdiv(已算 + 这步要算, 16) − 名下已有的块**，从 waiting 进 running 那一次，命中块还没挂到名下，所以还要再减掉它们。gen1 得 2。

然后 kv_cache_manager.py:376 做一个比较：`2 > 33757`？不是，继续。D3 里 req3 被抢占，就是这行返回 None 的那一刻。

33757 这个数：启动日志里 "GPU KV cache size: 540,128 tokens"，除以 16 是 33758 块；block 0 是 null block，永远不发出去，所以可用 33757。

## 第二段：拿块

```python
# kv_cache_manager.py:393-398
new_blocks = self.coordinator.allocate_new_blocks(request.request_id, num_tokens_need_slot, ...)
```

single_type 那边又算一次减法，这次减的是名下已有的块：

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
if num_new_blocks <= 0:
    return []
new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
req_blocks.extend(new_blocks)
```

`block_pool.get_new_blocks`（block_pool.py）从 free 队列队头 `popleft_n`，每块 `ref_cnt += 1`。刚启动时队列就是 1、2、3…顺序排着，所以 gen1 拿到 `[1, 2]`，两块 `ref_cnt=1`、`block_hash=None`。

`req_to_blocks[request_id]` 是这条请求 block table 的唯一真身，以后传给 model runner 的 `block_ids` 就从这里读。

decode 阶段这个减法大多数时候等于 0。第一个 decode step：computed 25，要算 1，need_slot 26，cdiv 还是 2，名下已有 2，不拿。一直到 computed 32、need_slot 33，cdiv 变 3，才拿第 3 块。中间 26 到 32 那 7 步 KV cache manager 什么都没做。块是按需一次一块追加的，不预留。

gen1 全程的节奏：

| computed | 名下块 | 这一步登记的 hash |
|---|---|---|
| 0 | [1, 2] | block 1 |
| 31 | 不拿 | block 2 |
| 32 | + 3 | |
| 47 | 不拿 | block 3 |
| 48 | + 4 | |
| …… | | |
| 79 | 不拿 | block 5 |
| 80 | + 6 | |
| 88（结束） | 6 块 | 共 5 条；num_tokens=89，block 6 只有 9 个 token，不满 |

## 第三段：登记 hash

```python
# kv_cache_manager.py:410-414
num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

gen1 进门时 `num_tokens_to_cache = min(0 + 25, 25) = 25`，25 // 16 = 1 个满块，block 1 的 hash 登记进 `cached_block_hash_to_block`。调试器里 `len(cached_block_hash_to_block)` 从 0 变 1。

**此刻还在 `schedule()` 里，第一个 forward 没跑，block 1 里一个 KV 值都没有。** 登记的是 hash，hash 只依赖 token id 和前一块的 hash，而 token id 早就知道了（block_pool.py 里的注释：hash 在 Request 建好和追加 token 时算好，存在 `request.block_hashes`）。等别的请求来命中它时，forward 早已跑完，KV 已经写进去。

single_type:286-301 记着每条请求已登记几块（`num_cached_block`），只登记新满的那几块，`block_pool.cache_full_blocks`（block_pool.py）给每块挂 hash 再插进 map。

## gen2：命中了，为什么还要 2 块

同一条 prompt 第二次进来。scheduler 先问 `get_computed_blocks`（kv_cache_manager.py:183-223）：

```python
max_cache_hit_length = request.num_tokens - 1      # 24
computed_blocks, num_new_computed_tokens = self.coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
```

命中 16 个 token，一块。cache 里明明有 5 条，为什么只命中 1 块？两个理由叠着：

- 只命中了一个 hash 值。`request.block_hashes` 按这条请求自己的 token 算，25 个 token 只凑满 1 块。cache 里另外 4 条是 gen1 第 17 到 80 个 token 的 hash，混着 gen1 的输出，新请求没那些 token。
- 上限 24 也只够 1 块。就算 prompt 有 32 个 token，`max_cache_hit_length = 31`，31 // 16 = 1。最后一个 token 必须留着重算，要它的 logits 才能采样下一个。

于是 allocate_slots 进门：comp 0、new_comp 16、new 9。`num_tokens_need_slot` 还是 25，cdiv 还是 2。命中 1 块，按理只差 1 块。可 `num_blocks_to_allocate` 报 **2**，回头看 single_type:164：

```python
num_evictable_blocks = self._get_num_evictable_blocks(new_computed_blocks[...])   # ref_cnt == 0 的命中块数
return num_new_blocks + num_evictable_blocks                                       # 1 + 1
```

block 1 此刻 `ref_cnt = 0`。gen1 结束时 free 把它放回了 free 队列，hash 还挂着。"free" 在这里的定义是"在 free 队列里"，不是"没人用"，`get_num_free_blocks()` 数的就是队列长度。

gen2 认领 block 1 走 `allocate_new_computed_blocks`（single_type:216-218）调 `block_pool.touch`（block_pool.py:399-404）：

```python
if block.ref_cnt == 0 and not block.is_null:
    self.free_block_queue.remove(block)
block.ref_cnt += 1
```

从队列摘走一个，队列长度减 1。再 `popleft` 一个新块给第 17 到 25 个 token，又减 1。所以 free 从 33757 到 33755，但只有 1 块是新的。

那多算的 1 块不是保守，是如实。假设 free 队列只剩 1 个，而那 1 个恰好是 block 1：gen2 认领之后队列空了，第 17 到 25 个 token 没块可拿，`get_new_blocks` 会在 block_pool.py:334 抛 ValueError。把命中的 evictable 块算进需求，`2 > 1` 直接返回 None，请求留在 waiting，不会撞到异常。

新块是 **7**，不是 2。gen1 结束时 free 把 `[6, 5, 4, 3, 2, 1]` 倒序追加到队尾，队头还是从没用过的 7。倒序是故意的：尾块先被 evict，前缀块活得久。

为什么不直接拿 block 2？它装着 gen1 第 17 到 32 个 token，可 gen2 此刻只有 25 个 token，算不出那一块的 hash。而且那块后 7 个是 gen1 的输出，gen2 能不能生成一样的还不知道。

## 5 变 6：temperature 0 不保证 bitwise 一致

gen2 跑完，`cached_block_hash_to_block` 从 5 条变 6 条。同一条 prompt、temperature 0，按理 64 个输出一模一样，5 条 hash 一条不多。

通过对比两次的 token id：gen2 从第 76 个 token 起和 gen1 不一样。gen1 是 "reflecting the sunlight and causing it to bend"，gen2 是 "reflecting and refracting the sunlight"。第 76 个落在第 5 块（65 到 80），第 5 块的 hash 变了，多 1 条。

原因是 batch 形状。gen1 的 prefill 是 25 个 token 一起算，gen2 命中 16 个后只算 9 个，kernel 的归约顺序不同，logits 有浮点级差异。前 75 步差异没翻转 argmax，第 76 步翻了。temperature 0 保证每步取最大，不保证跨形状 bitwise 一致。

顺带一个下一篇的伏笔。前 4 块 hash 相同，但 vLLM 不去重：同一条 hash 下挂着两个 block，`[2, 7]`、`[3, 8]`、`[4, 9]`。block_pool.py:48-52 的 NOTE #1 说明原因，block table 只追加，不改已发出去的 block id。

## main 对照

`git diff v0.21.0..main -- vllm/v1/core/kv_cache_manager.py`（main @ 83fe99399e，2026-09-09），+438 行。

- **watermark**（PR #44594，2026-06-11）：`required_blocks = num_blocks_to_allocate + watermark_blocks`，只对 waiting/preempted 的请求生效。给 free 留一截水位，免得刚准入 free 就归零，下一步多要一块就立刻抢占。默认 0，行为不变。
- **reserved_blocks**（main :539）：`available_blocks = free − reserved_blocks`，给 async KV connector 用，防止它的初始分配吃掉正在 prefill 的请求依赖的块。
- **remove_skipped_blocks 改按 processed 基准**（PR #47728，2026-07-08）：传的是 `total_computed_tokens − request.num_in_flight_tokens`。async scheduling 下 in-flight 的 step 还在读窗口下面的块，不能按调度区的记录，得按真实计算的来。和 D4 的 placeholder 是同一根线。
- `get_computed_blocks` 返回三元组，多了 `shared_prefix_boundary`（PR #47782），给 Mamba / sliding window 这类不保留全部前缀的组用，它回答的是"各类 cache group 都能当共同前缀用的边界在哪"，full attention 恒为 0。

## 小结

- 要几块 = cdiv(已算 + 这步要算, 16) − 名下已有，从 waiting 进入 running 时再减命中块
- hash 在分配时就登记，模型还没算；登记数封顶在 `request.num_tokens`，placeholder 不算
- 命中块 ref_cnt=0 时躺在 free 队列里，认领它也消耗一个 free 名额，所以要算进需求
- free 队列 FIFO，归还倒序进队尾，尾块先被 evict，前缀块活得久
- temperature 0 不等于跨 batch 形状 bitwise 一致，prefix cache 命中会改变 prefill 形状

---

下一篇进 block_pool：free 队列的数据结构、hash 怎么算、什么时候一条 hash 真的被删。

</div>

<div class="lang lang-en" markdown="1">

This post is based on vLLM v0.21.0 (V1 engine, offline `LLM.generate` path, multiprocessing off, prefix caching on by default). With a debugger I trace `KVCacheManager.allocate_slots()`: the scheduler says "compute N tokens this step", and someone has to turn N into "how many blocks, which blocks, and which of them get registered in the prefix cache". Line numbers refer to that version.

The previous three posts stayed inside the scheduler and looked at "how much to compute this step". Every time the scheduler comes up with a `num_new_tokens`, it asks the KV cache manager once: are there enough blocks? If yes, take them; if not, preempt someone or wait. This post looks at the side being asked.

The configuration goes back to real block counts: `max_model_len=2048`, `gpu_memory_utilization=0.1`, no more 10-block cap. The three generates are the same as before: gen1 cold start / gen2 same prompt / gen3 two prompts concurrently.

A few numbers from the debugger, up front:

- The block pool has **33758** blocks, **33757** usable
- gen1 needs **2** blocks to go from waiting to running and gets `[1, 2]`; in the same call, block 1's hash is registered in the prefix cache, and the model has not run a single step
- gen2 hits block 1 yet still reports "need 2 blocks"; free drops by 2, but `new_blocks` is only `[7]`
- After gen2 finishes, the cache goes from 5 entries to **6**, same prompt, temperature 0

## Where: three layers, one function

```
scheduler.schedule()
├── :702  waiting section  allocate_slots(request, num_new_tokens, num_new_computed_tokens=…, new_computed_blocks=…, …)
└── :425  running section  allocate_slots(request, num_new_tokens, num_lookahead_tokens=…)
        └── KVCacheManager.allocate_slots()          kv_cache_manager.py:225
              └── KVCacheCoordinator                 kv_cache_coordinator.py   forwards once per kv_cache_group
                    └── SingleTypeKVCacheManager     single_type_kv_cache_manager.py   the actual block arithmetic
                          └── BlockPool              block_pool.py   free queue + cached_block_hash_to_block
```

The Qwen2.5-0.5B I use is all full attention, so there is a single kv_cache_group, the coordinator is `UnitaryKVCacheCoordinator`, and under it sits one `FullAttentionManager`. The coordinator layer can be treated as transparent today; it is just `for manager in self.single_type_managers` forwarding.

The two call sites pass different arguments. The waiting-section call carries `new_computed_blocks` (which blocks hit the prefix cache); the running-section call only says "how many to compute this step". The first thing the KV cache manager does is fold them into two numbers:

```python
num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
total_computed_tokens = min(num_local_computed_tokens + num_external_computed_tokens, self.max_model_len)
```

The layout diagram in the docstring (:262-294) is worth a look. It slices a request's token sequence into five segments:

```
Blocks layout:
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                          |            < to be allocated >           |
----------------------------------------------------------------------
                          | < to be cached (roughly, |
                          | details below)>          |
----------------------------------------------------------------------
| Prefix-cached tokens from either vLLM   |
| or connector. Can be safely removed if  |
| they are outside sliding window.        |
----------------------------------------------------------------------
|   < cached by vLLM >    | not cached by |
                          | vLLM, but     |
| ref_cnt  | ref_cnt not  | cached by     |
| increased| increased yet| connector     |
----------------------------------------------------------------------
```

- **comp**: computed in earlier steps; the blocks already exist.
- **new_comp**: just hit in the prefix cache this time; the blocks are in the cache and need to be attached to this request.
- **ext_comp**: fetched externally by a KV connector; we have none, always 0.
- **new**: tokens to compute this time; blocks must be newly taken.
- **lookahead**: draft slots for spec decode; ours is 0.

The three rows under the diagram say which segments the function's three actions cover:

- "to be computed" = new + lookahead
- "to be allocated" = new_comp + ext_comp + new + lookahead, i.e. everything to the right of comp
- "to be cached" ≈ up to the end of new, excluding lookahead, because drafts may be rejected

**Block 1's hash is registered in the prefix cache before the model computes anything.** We are still inside `schedule()`; the first forward has not run. The hash depends only on the token ids and the previous block's hash. It is computed when the Request is built and whenever tokens are appended, stored in `request.block_hashes`, and does not depend on KV values. The KV values are written by this step's forward; by the time another request hits the block, that forward has long finished.

gen1 on entry: comp 0, new_comp 0, new 25. gen2 on entry: comp 0, new_comp 16, new 9. The whole post is about how these two sets of numbers turn into blocks.

## Stage one: how many blocks

```python
# kv_cache_manager.py:351-353
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, self.max_model_len)
```

gen1 on entry: `num_tokens_need_slot = 0 + 25 = 25`. Then it asks the coordinator how many blocks. The real arithmetic lives in single_type_kv_cache_manager:

```python
num_required_blocks = cdiv(num_tokens, self.block_size)        # cdiv(25, 16) = 2
num_req_blocks = len(self.req_to_blocks.get(request_id, ()))   # 0, no blocks owned yet

if request_id in self.num_cached_block:                        # running requests take this fast path
    return max(num_required_blocks - num_req_blocks, 0)

num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks   # hit blocks + owned blocks
num_new_blocks = max(num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0)
num_evictable_blocks = self._get_num_evictable_blocks(new_computed_blocks[...])
return num_new_blocks + num_evictable_blocks
```

In one sentence: **blocks needed = cdiv(computed + to compute this step, 16) − blocks already owned**. On the one call that moves a request from waiting to running, the hit blocks are not yet attached, so they are subtracted as well. gen1 gets 2.

Then kv_cache_manager.py:376 makes one comparison: `2 > 33757`? No, carry on. In D3, req3 getting preempted was exactly the moment this line returned None.

About 33757: the startup log says "GPU KV cache size: 540,128 tokens", divided by 16 that is 33758 blocks; block 0 is the null block and is never handed out, so 33757 are usable.

## Stage two: taking blocks

```python
# kv_cache_manager.py:393-398
new_blocks = self.coordinator.allocate_new_blocks(request.request_id, num_tokens_need_slot, ...)
```

single_type does the subtraction once more, this time against the blocks the request already owns:

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
if num_new_blocks <= 0:
    return []
new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
req_blocks.extend(new_blocks)
```

`block_pool.get_new_blocks` (block_pool.py) does `popleft_n` from the head of the free queue and bumps `ref_cnt += 1` on each block. Right after startup the queue is simply 1, 2, 3… in order, so gen1 gets `[1, 2]`, both with `ref_cnt=1` and `block_hash=None`.

`req_to_blocks[request_id]` is the single source of truth for this request's block table; the `block_ids` later handed to the model runner are read from here.

During decode this subtraction is 0 most of the time. First decode step: computed 25, compute 1, need_slot 26, cdiv still 2, 2 owned, nothing taken. Not until computed 32, need_slot 33, cdiv becomes 3, is the third block taken. For the 7 steps from 26 to 32 the KV cache manager does nothing. Blocks are appended one at a time on demand, nothing is reserved.

gen1's rhythm from start to finish:

| computed | owned blocks | hash registered this step |
|---|---|---|
| 0 | [1, 2] | block 1 |
| 31 | none taken | block 2 |
| 32 | + 3 | |
| 47 | none taken | block 3 |
| 48 | + 4 | |
| …… | | |
| 79 | none taken | block 5 |
| 80 | + 6 | |
| 88 (finished) | 6 blocks | 5 entries total; num_tokens=89, block 6 holds only 9 tokens, not full |

## Stage three: registering hashes

```python
# kv_cache_manager.py:410-414
num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

gen1 on entry: `num_tokens_to_cache = min(0 + 25, 25) = 25`, 25 // 16 = 1 full block, so block 1's hash goes into `cached_block_hash_to_block`. In the debugger, `len(cached_block_hash_to_block)` goes from 0 to 1.

**We are still inside `schedule()`, the first forward has not run, and block 1 contains not a single KV value.** What gets registered is a hash. The hash depends only on the token ids and the previous block's hash, and the token ids are known long in advance (the comment in block_pool.py: hashes are computed when the Request is built and when tokens are appended, stored in `request.block_hashes`). By the time another request hits it, the forward has long finished and the KV has been written.

single_type:286-301 tracks how many blocks each request has registered (`num_cached_block`) and only registers the newly filled ones; `block_pool.cache_full_blocks` (block_pool.py) attaches the hash to each block and inserts it into the map.

## gen2: it hit, so why still 2 blocks

The same prompt comes in a second time. The scheduler first asks `get_computed_blocks` (kv_cache_manager.py:183-223):

```python
max_cache_hit_length = request.num_tokens - 1      # 24
computed_blocks, num_new_computed_tokens = self.coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
```

16 tokens hit, one block. The cache clearly has 5 entries, so why only 1 block? Two reasons stacked together:

- Only one hash matched. `request.block_hashes` is computed from this request's own tokens, and 25 tokens fill only 1 block. The other 4 entries in the cache are hashes of gen1's tokens 17 to 80, mixed with gen1's output; the new request does not have those tokens.
- The cap of 24 also allows only 1 block. Even if the prompt had 32 tokens, `max_cache_hit_length = 31`, and 31 // 16 = 1. The last token must be recomputed, because its logits are needed to sample the next one.

So allocate_slots is entered with comp 0, new_comp 16, new 9. `num_tokens_need_slot` is still 25, cdiv still 2. With 1 block hit, it should be short only 1. Yet `num_blocks_to_allocate` reports **2**. Look back at single_type:164:

```python
num_evictable_blocks = self._get_num_evictable_blocks(new_computed_blocks[...])   # hit blocks with ref_cnt == 0
return num_new_blocks + num_evictable_blocks                                       # 1 + 1
```

Block 1 has `ref_cnt = 0` at this point. When gen1 finished, free put it back on the free queue with its hash still attached. "Free" here means "in the free queue", not "unused"; `get_num_free_blocks()` simply counts the queue length.

gen2 claims block 1 via `allocate_new_computed_blocks` (single_type:216-218), which calls `block_pool.touch` (block_pool.py:399-404):

```python
if block.ref_cnt == 0 and not block.is_null:
    self.free_block_queue.remove(block)
block.ref_cnt += 1
```

One block is pulled out of the queue, length minus 1. Then a `popleft` for a new block to hold tokens 17 to 25, minus 1 again. So free goes from 33757 to 33755, but only 1 block is new.

The extra block in the count is not conservatism, it is accuracy. Suppose the free queue had only 1 block left, and that block happened to be block 1: after gen2 claims it the queue is empty, tokens 17 to 25 have nowhere to go, and `get_new_blocks` raises ValueError at block_pool.py:334. Counting the hit evictable blocks into the demand makes `2 > 1` return None directly, the request stays in waiting, and the exception is never reached.

The new block is **7**, not 2. When gen1 finished, free appended `[6, 5, 4, 3, 2, 1]` to the tail in reverse order, and the head is still the never-used 7. The reverse order is deliberate: tail blocks get evicted first, prefix blocks live longer.

Why not just take block 2? It holds gen1's tokens 17 to 32, but gen2 has only 25 tokens at this point and cannot compute that block's hash. Besides, the last 7 tokens of that block are gen1's output, and whether gen2 will generate the same ones is not yet known.

## 5 to 6: temperature 0 does not guarantee bitwise equality

After gen2 finishes, `cached_block_hash_to_block` goes from 5 entries to 6. Same prompt, temperature 0; in principle the 64 outputs should be identical and the 5 hashes should gain nothing.

Comparing the token ids of the two runs: gen2 diverges from gen1 at token 76. gen1 says "reflecting the sunlight and causing it to bend", gen2 says "reflecting and refracting the sunlight". Token 76 falls in block 5 (65 to 80), so block 5's hash changes, one entry more.

The cause is batch shape. gen1's prefill computes 25 tokens together; gen2 hits 16 and computes only 9, so the kernel's reduction order differs and the logits carry floating-point-level differences. For 75 steps the difference did not flip the argmax; at step 76 it did. Temperature 0 guarantees taking the max at each step; it does not guarantee bitwise equality across shapes.

A side note that foreshadows the next post. The first 4 blocks share hashes, but vLLM does not deduplicate: the same hash has two blocks hanging under it, `[2, 7]`, `[3, 8]`, `[4, 9]`. NOTE #1 at block_pool.py:48-52 explains why: block tables are append-only, and already-issued block ids are never changed.

## Comparison with main

`git diff v0.21.0..main -- vllm/v1/core/kv_cache_manager.py` (main @ 83fe99399e, 2026-09-09), +438 lines.

- **watermark** (PR #44594, 2026-06-11): `required_blocks = num_blocks_to_allocate + watermark_blocks`, applied only to waiting/preempted requests. It leaves some headroom in free so a freshly admitted request does not drive free to zero and trigger a preemption the moment it needs one more block. Default 0, behavior unchanged.
- **reserved_blocks** (main :539): `available_blocks = free − reserved_blocks`, for async KV connectors, to stop their initial allocation from eating blocks that requests currently prefilling depend on.
- **remove_skipped_blocks now uses a processed baseline** (PR #47728, 2026-07-08): it is passed `total_computed_tokens − request.num_in_flight_tokens`. Under async scheduling the in-flight steps are still reading blocks below the window, so it must go by what has actually been computed rather than the scheduler's records. Same thread as the placeholders in D4.
- `get_computed_blocks` now returns a triple with an extra `shared_prefix_boundary` (PR #47782), for groups like Mamba / sliding window that do not keep the full prefix. It answers "where is the boundary that every cache group can treat as a shared prefix"; for full attention it is always 0.

## Summary

- Blocks needed = cdiv(computed + to compute this step, 16) − blocks owned; on the waiting-to-running transition, subtract hit blocks too
- Hashes are registered at allocation time, before the model computes; the count is capped at `request.num_tokens`, placeholders do not count
- A hit block with ref_cnt=0 sits in the free queue; claiming it also consumes one free slot, so it must be counted into the demand
- The free queue is FIFO; returns go to the tail in reverse order, tail blocks get evicted first, prefix blocks live longer
- Temperature 0 does not mean bitwise equality across batch shapes, and a prefix cache hit changes the prefill shape

---

Next post goes into block_pool: the free queue's data structure, how hashes are computed, and when a hash entry is actually deleted.

</div>
