---
title: 'vLLM 源码追踪（三）· Scheduler 中：谁能进来，谁得让位'
title_en: 'Tracing vLLM Source (3) · Scheduler II: Who Gets In, Who Gives Way'
date: 2026-09-08
permalink: /posts/2026/09/vllm-scheduler-2/
tags:
  - LLM
  - vLLM
  - 源码追踪
---

<div class="lang lang-zh" markdown="1">

本文基于 vLLM v0.21.0（V1 引擎，离线 `LLM.generate` 路径，关闭 multiprocessing，async scheduling 默认开启），用 debugger 追踪 `schedule()` 的 waiting 段：request 怎么被放进来、prefix cache 怎么命中、block 不够时谁被抢、被抢的怎么回来。文中行号以该版本为准。

书接上回。上篇把 `max_num_batched_tokens` 压到 16 看 token 不够用；这次换个方向，把 **KV block 压到 9 个**看显存不够用。同一条 25-token 的 prompt 跑三次，调试器里出现了三种结果：

- 第一次：算 25 个
- 第二次：算 **9** 个（命中了 16）
- 第三次和另一条 prompt 一起跑：另一条跑到第 64 个 token 被**抢占**，回来时 65 个 token 只重算 **17** 个

配置：

```python
max_model_len=128
max_num_seqs=2
num_gpu_blocks_override=10   # block 0 是 null block，可用 9 个，每块 16 token
# max_num_batched_tokens 不再压，今天不切段
```

三次 `generate`：

```
gen1 = PROMPT                 冷启动
gen2 = 同一条 PROMPT
gen3 = [PROMPT, PROMPT_B]     PROMPT_B 24 token
```

每条都生成到 64 个 token 封顶。

## waiting 段在哪

上篇我们一直在 `schedule()` 的 **running 段**。今天的所有断点都在它后面的 **waiting 段**：

```
schedule()
├── running 段：已经在跑的 request，每条该算多少          ← 上篇
└── waiting 段：running 段用剩的 budget，放新 request 进来  ← 本篇
      ├── get_computed_blocks   → prefix cache 命中多少
      ├── min(num_new_tokens, token_budget)
      ├── allocate_slots        → 拿不到 block 就 break
      └── running.append        → WAITING 进 scheduled_new_reqs / PREEMPTED 进 scheduled_resumed_reqs
```

顺序很重要：**先 running 后 waiting**，所以正在跑的永远比排队的优先，新 request 只能吃剩下的 budget。上篇 step 2 剩下的那 7 个 token，就是留给这一段的。

问题一个一个来。

## 准入：gen1，一条冷 request 怎么进 running

waiting 段每条 request 要过三站。

**Step 1 · 问 prefix cache**：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

gen1 是冷启动的：

```
num_new_local_computed_tokens = 0
get_block_ids() = ([],)
request.status = WAITING
```

顺便看到 `len(request.block_hashes) = 1`：25 个 token 只有 1 条 hash，这个数字下一节要用。

**Step 2 · 算这次能调度多少**：

```python
num_new_tokens = request.num_tokens - num_computed_tokens   # 25 - 0
...
num_new_tokens = min(num_new_tokens, token_budget)          # min(25, 256) = 25
assert num_new_tokens > 0
```

这行 `min()` 和上篇 running 段的那行长得一模一样。区别只在**减数**：running 段减的是 `num_computed_tokens`（已经算到哪了），这里减的是 prefix cache 给的 `num_computed_tokens`（不用算的有多少）。**表达式同、位置异、对象状态不同**：一个是 RUNNING 的 request 追进度，一个是 WAITING/PREEMPTED 的 request 定起点。

注释里特意说了用 `request.num_tokens` 而不是 `num_prompt_tokens`，是为了被抢占后回来的 request。它已经有 output token 了。这条伏笔到 gen3 才用上。

**Step 3 · 要 block，进 running**：

```python
new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
if new_blocks is None:
    break                              # 拿不到 block：今天不进，留在 waiting
...
self.running.append(request)
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
...
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

gen1 拿到 `new_blocks = ([1, 2],)`，free block 9 → 7（ceil(25/16) = 2）。status 变 RUNNING，之后的 step 它就走上篇的 running 段了。

> 准入 = running 段用剩的 budget + 拿得到 block，两个条件缺一不可。

## prefix cache：gen2，为什么是 16 不是 25

gen2 同一条 prompt。step 1 停下：

```
num_new_local_computed_tokens = 16
new_computed_blocks.get_block_ids() = ([1],)
```

step 2：`num_new_tokens = 25 − 16 = 9`。step 3 之后 `num_computed_tokens → 16`，这条 request 生下来就"算到 16 了"。

一开始我以为命中会是 25，因为 gen1 明明把 25 个 token 的 KV 都算过了。

**实际钥匙是 hash，hash 只给满块算**：25 = 16 + 9，第一块满 16 有 hash，第二块只有 9 个 token 没有 hash，所以 gen1 时 `len(block_hashes) = 1`。没满一个块就查不到，不管 KV 在不在。

另一个反直觉的点：Debug Console 里敲 `new_computed_blocks.blocks[0][0].ref_cnt` 是 **0**。gen1 结束时 block 1 已经 free 了，ref_cnt 归零，但它的 hash 还留在 `cached_block_hash_to_block` 里。**free 只是还所有权，不清 hash 表。**这也是下一节抢占能"捡回来"的原因。

命中判定一共三层，都在 kv_cache_manager 那边：

```
kv_cache_manager.get_computed_blocks         (kv_cache_manager.py)
  └── coordinator.find_longest_cache_hit     沿 block_hashes 前缀链逐个查，断了就 break
        └── block_pool.get_cached_block      dict 查一下
```

其中，有一行：

```python
max_cache_hit_length = request.num_tokens - 1
```

为什么最多只让命中 `num_tokens − 1`？全命中就意味着这个 step 一个 token 都不用算，没有 logits 可 sample；而且后面有一句 `assert num_new_tokens > 0` 会直接炸。所以哪怕全部命中也硬留最后一个 token 重算——注释还承认这可能白算一整块（因为 `num_computed_tokens` 要对齐 block）。

> prefix cache 命中 = 沿 block hash 前缀链能查到多少个满块，命中数直接写进 num_computed_tokens。

## 抢占：gen3，谁被踢、丢了什么、怎么回来

gen3 两条一起：req 2（PROMPT）命中 16 → 算 9，req 3（PROMPT_B，24 token）命中 0 → 算 24。进 running 之后 `len(self.running) = 2`，free block 5。

两条各自 decode。9 个块装不下两条 128 长的 request——`max_model_len=128`、10 块。于是某一步 running 段里 req 3 要新块，`allocate_slots` 返回 None，进了 scheduler.py:424–468 的抢占循环。

不是：

```
发现未来可能不够
→ 提前拒绝 req 3
```

而是：

```
req 2、req 3 都先跑
→ KV 随 decode 动态增长
→ 真正用光
→ 某条申请新 block 失败
→ 从 running 里踢一个出去
→ 腾 block
→ 被踢的以后 recompute 回来
```

### 谁被踢

```python
preempted_req = self.running.pop()
self._preempt_request(preempted_req, scheduled_timestamp)
preempted_reqs.append(preempted_req)
if preempted_req == request:
    break
```

FCFS 策略下就是 `running.pop()`：**队尾、最新进来的那条**。这里 free = 0 时缺块的是 req 3，队尾也是 req 3——它踢的是自己。`if preempted_req == request: break` 就是给这种情况的：自己都被踢了，没人可再踢，这个 step 排不上了。

我一开始以为被抢的总是"别人"。看到这行才反应过来：抢占是为了腾块，腾出来的块谁用没规定；这次腾出来的 4 块直到 req 2 finish 都一直空着。

### 丢了什么

`_preempt_request` 逐行看：

```python
self.kv_cache_manager.free(request)      # free block 0 → 4（64 token 正好 4 块）
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0          # 64 → 0
request.num_preemptions += 1
self.waiting.prepend_request(request)    # 回 waiting 队头，不是队尾
```

进门时 `num_computed_tokens = 64`、`num_tokens = 64`；出门 `num_computed_tokens = 0`，但 `num_tokens` 还是 64。**归零的是"算到哪了"，不是"有多少 token"**——24 个 prompt + 40 个 output 都还在 request 对象里。

### 怎么回来

下一次 `schedule()`，waiting 段队头就是 req 3。

```
status = PREEMPTED
num_tokens = 65
num_new_local_computed_tokens = 48
new_computed_blocks.get_block_ids() = ([3, 2, 6],)
```

刚 free 掉的 3 个满块，hash 还在表里，被 prefix cache 原样捡回来。Step 2：`num_new_tokens = 65 − 48 = 17`。这就是上面那条注释说的"用 `num_tokens` 是为了 resumed request"——被抢的回来时，prompt 和已经生成的 output 一视同仁，都是"需要算到"的 token。

然后 Step 3 卡住了：`allocate_slots` 返回 None，`break`。要 3 个 cache 块 + 2 个新块 = 5，free 只有 4。这里连停三次，每次 `get_num_free_blocks()` 都是 4，req 3 就躺在 waiting 队头，req 2 继续 decode。

req 2 到 64 finish，块释放，再进 waiting 段：站 3 这次 `request.status == PREEMPTED` → 进 **`scheduled_resumed_reqs`**，`num_computed_tokens → 48`，只重算 17 个。

### 和论文的 recompute 不是一回事

PagedAttention 论文里的 recompute 是全丢全算：回来重算 65 个。v1 里 `free()` 只还所有权、不清 hash，回来先过一遍 prefix cache，只重算没满块的尾巴。**丢的是 KV 的所有权，不是内容**——只要块没被别人覆盖，就能捡回来。

### 一个没来得及登记的 Block

req 3 最终被抢占后有 65 个 token。`block_size=16`，按理说前 64 个 token 正好对应 4 个完整 block。但它恢复时，prefix cache 却只捡回了 3 个：

```
num_new_local_computed_tokens = 48
new_computed_blocks.get_block_ids() = ([3, 2, 6],)
```

为什么第 4 个满块没有回来？

我一开始以为是被 evict 了。结果不是。

给 `cache_blocks()` 加了断点后才发现，真正的原因是：

> **第 4 个 block 已经填满，hash 也已经生成，但还没来得及登记进 prefix cache，req 3 就被抢占了。**

这里最重要的是区分：

```
block 已经填满
block hash 已经生成
block 已经登记进 prefix cache
```

这是三个不同的时刻。

#### 第一步：63 个 token，只能登记 3 个完整 block

抢占前一拍，schedule 进入 `allocate_slots()` 时：

```
num_tokens = 63
len(block_hashes) = 3
```

此时 token 分布是：

```
[ 16 ][ 16 ][ 16 ][ 15 ]
```

所以只有前三个 block 是完整的。`allocate_slots()` 最后执行：

```
cache_blocks(63)
```

prefix cache 中成功登记了前三块。

#### 第二步：token 64 回来，第 4 块填满，但还没登记

随后 GPU 返回一个 output token：

```
num_tokens: 63 → 64
len(block_hashes): 3 → 4
```

现在已经变成：

```
[ 16 ][ 16 ][ 16 ][ 16 ]
```

因此第 4 个 block 的 hash 已经可以算出来。

但是 async scheduler 此时调用的是：

```python
self.kv_cache_manager.cache_blocks(
    request,
    request.num_computed_tokens - request.num_output_placeholders,
)
```

这一时刻的状态相当于：

```
num_computed_tokens       = 64
num_output_placeholders   = 1
                           ----
可登记的 cache frontier   = 63
```

所以它仍然执行：

```
cache_blocks(63)
```

而不是 `cache_blocks(64)`。

也就是说，此刻第 4 块的状态是：

```
KV 已经存在          ✓
block hash 已生成    ✓
prefix cache 已登记  ✗
```

这就是后面"漏掉一块"的根源。

#### 第三步：下一拍本来应该把第 4 块补登记

正常情况下，下一拍 schedule 再次进入 `allocate_slots()` 后，会计算：

```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
```

此时本来可以得到：

```
min(64 + 1, 64) = 64
```

于是走到后面的：

```
cache_blocks(64)
```

第 4 个 block 就会正式进入 prefix cache。

但这次恰好：

```
free blocks = 0
```

`allocate_slots()` 在前面的容量检查就直接返回了：

```
抢占这一拍
allocate_slots()
    ↓
num_blocks_to_allocate > free
    ↓
return None
```

因此，本该执行的 `cache_blocks(64)` 根本没有发生。

接下来 scheduler 立刻抢占 req 3：

```
allocate_slots() → None
        ↓
_preempt_request(req 3)
        ↓
释放 req 3 占用的 KV blocks
        ↓
computed progress 归零
        ↓
req 3 回到 waiting
```

所以第 4 个 block 从始至终都没有被登记进 prefix cache。

#### 第四步：为什么最后看到的是 65 个 token？

抢占发生时 req 3 有 64 个 token，但此前还有一次已经发给 GPU 的计算在 in-flight，抢占不会取消它。所以抢占之后这个 output 照样回来、照样 append，调试器里才会看到 `status = PREEMPTED`、`num_tokens = 65`。

这个 65 是抢占之后才出现的状态，不能倒过来代入前面那次 `64 − 1 = 63` 的 `cache_blocks()` 调用。把四个时刻排开看：

```
① 抢占前一拍
   num_tokens = 63
   → cache_blocks(63)
   → 登记 3 块

② token 64 返回
   num_tokens = 64
   block_hashes = 4
   → async cache_blocks(64 - 1)
   → 仍然只登记到 63
   → 第 4 块有 hash，但没登记

③ 抢占这一拍
   本应 cache_blocks(64)
   但 free = 0
   → allocate_slots() 提前 return None
   → req 3 被抢占

④ 抢占后
   之前的 in-flight output 返回
   num_tokens = 65
```

所以最终恢复时，prefix cache 只能找到曾经真正登记过的前三块：

```
new_computed_blocks.get_block_ids() = ([3, 2, 6],)
```

这不是 eviction。第 4 个 block 根本没有经历：

```
登记 → 被淘汰
```

它经历的是：

```
填满
→ hash 生成
→ 尚未登记
→ 下一拍分配失败
→ preempt
→ 错过登记机会
```

因此，这里最值得记住的是：

> **hash 生成 ≠ prefix cache 登记。async scheduling 让二者之间存在一个短暂的时间差；这次 preemption 恰好发生在这个窗口里，于是一个已经填满、已经有 hash 的 block，最终仍无法在恢复时被 prefix cache 找回来。**

抢占与恢复的整个过程也可以压成两行：

```
抢占：free KV blocks + computed progress reset + 回 waiting
恢复：查 prefix cache → 捡回已登记 block → 其余 recompute
```

## 题外话：main 上这一段变了吗

上篇提到 main 把 running 段的 `min()` 变成三参数 `min(num_new_tokens, token_budget, input_budget − draft_slots)`。waiting 段完全对称：入口多了 `if input_budget <= draft_slots: break`，`min()` 同样变三参，准入后 `input_budget -= num_new_tokens + draft_slots`。`max_cache_hit_length = num_tokens − 1` 和 `num_tokens_to_cache` 两处封顶原样保留。机制没变，只是 budget 多了一维。

## 小结

- 准入 = running 段用剩的 budget + 拿得到 block
- prefix cache 命中 = 沿 block hash 前缀链查到几个满块，命中数直接成为 num_computed_tokens
- 抢占 = FCFS 下 `running.pop()` 踢最新的，可能踢到自己；归零的是 num_computed_tokens，不是 num_tokens
- 被抢的回来 = 先捡 prefix cache，再当新 request 走一遍准入，进的是 scheduled_resumed_reqs

---

到这里，`schedule()` 里的两段调度逻辑就走完了。

但我们目前看到的还只是 scheduler 如何做决定：谁继续跑、谁拿 block、KV 不够时谁被抢占。真正执行完一拍之后，request 的状态还会继续变化。

前面已经看到一个例子：req 3 被抢占时还是 64 个 token，随后一个 in-flight output 返回，`num_tokens` 又变成了 65。但这个 token 到底在哪被写回 request？类似地：req 2 是在哪被判定为 finished 的？request 结束后，占用的 KV blocks 又是在哪真正还给 block pool 的？

这些都不发生在前面读过的调度决策逻辑里，而是在 GPU output 返回之后的 `update_from_output()` 这条路径上完成。下篇讲它。

</div>

<div class="lang lang-en" markdown="1">

This post is based on vLLM v0.21.0 (V1 engine, offline `LLM.generate` path, multiprocessing disabled, async scheduling on by default). Using a debugger, we trace the waiting section of `schedule()`: how a request gets admitted, how the prefix cache hits, who gets preempted when blocks run out, and how a preempted request comes back. Line numbers refer to that version.

Picking up from last time. The previous post squeezed `max_num_batched_tokens` down to 16 to watch tokens run short. This time we go the other way and **squeeze the KV blocks down to 9** to watch memory run short. The same 25-token prompt is run three times, and the debugger shows three different outcomes:

- First run: compute 25
- Second run: compute **9** (16 hit the cache)
- Third run, together with another prompt: the other one gets **preempted** at its 64th token, and when it comes back only **17** of its 65 tokens are recomputed

Configuration:

```python
max_model_len=128
max_num_seqs=2
num_gpu_blocks_override=10   # block 0 is the null block, so 9 usable, 16 tokens each
# max_num_batched_tokens is no longer squeezed; no chunking today
```

The three `generate` calls:

```
gen1 = PROMPT                 cold start
gen2 = the same PROMPT
gen3 = [PROMPT, PROMPT_B]     PROMPT_B is 24 tokens
```

Each request generates up to a cap of 64 tokens.

## Where the waiting section is

Last time we stayed in the **running section** of `schedule()`. Today every breakpoint sits in the **waiting section** that follows it:

```
schedule()
├── running section: requests already running, how much each computes    ← last post
└── waiting section: whatever budget running left over, admit new requests ← this post
      ├── get_computed_blocks   → how much the prefix cache hits
      ├── min(num_new_tokens, token_budget)
      ├── allocate_slots        → no block, then break
      └── running.append        → WAITING goes to scheduled_new_reqs / PREEMPTED goes to scheduled_resumed_reqs
```

The order matters: **running first, waiting second**, so requests already running always take priority over queued ones, and new requests only get what is left of the budget. The 7 tokens left over after step 2 in the last post were reserved for this section.

One question at a time.

## Admission: gen1, how a cold request enters running

Every request in the waiting section passes through three stations.

**Step 1 · Ask the prefix cache**:

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

gen1 is a cold start:

```
num_new_local_computed_tokens = 0
get_block_ids() = ([],)
request.status = WAITING
```

Along the way we see `len(request.block_hashes) = 1`: 25 tokens produce only 1 hash. Keep that number in mind for the next section.

**Step 2 · Work out how much to schedule this time**:

```python
num_new_tokens = request.num_tokens - num_computed_tokens   # 25 - 0
...
num_new_tokens = min(num_new_tokens, token_budget)          # min(25, 256) = 25
assert num_new_tokens > 0
```

This `min()` looks identical to the one in the running section from last post. The only difference is the **subtrahend**: the running section subtracts the request's own `num_computed_tokens` (how far it has computed), while this one subtracts the `num_computed_tokens` handed back by the prefix cache (how much does not need computing). **Same expression, different place, different object state**: one is a RUNNING request catching up on progress, the other is a WAITING/PREEMPTED request finding its starting point.

The comment explicitly says `request.num_tokens` is used instead of `num_prompt_tokens` for requests that come back after preemption. Those already have output tokens. This piece of foreshadowing pays off in gen3.

**Step 3 · Get blocks, enter running**:

```python
new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
if new_blocks is None:
    break                              # no block: not today, stay in waiting
...
self.running.append(request)
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
...
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

gen1 gets `new_blocks = ([1, 2],)`, and free blocks go 9 → 7 (ceil(25/16) = 2). Its status becomes RUNNING, and from the next step on it goes through the running section from last post.

> Admission = leftover budget from the running section + blocks available. Both conditions are required.

## Prefix cache: gen2, why 16 and not 25

gen2 sends the same prompt. Pausing at step 1:

```
num_new_local_computed_tokens = 16
new_computed_blocks.get_block_ids() = ([1],)
```

Step 2: `num_new_tokens = 25 − 16 = 9`. After step 3, `num_computed_tokens → 16`; this request is born "already computed up to 16".

At first I expected a hit of 25, since gen1 had clearly computed the KV for all 25 tokens.

**The real key is the hash, and hashes are only computed for full blocks**: 25 = 16 + 9. The first block is full at 16 and has a hash; the second has only 9 tokens and no hash. That is why gen1 showed `len(block_hashes) = 1`. A block that is not full cannot be looked up, regardless of whether its KV exists.

Another counterintuitive point: typing `new_computed_blocks.blocks[0][0].ref_cnt` in the Debug Console gives **0**. When gen1 finished, block 1 was freed and its ref_cnt dropped to zero, yet its hash stayed in `cached_block_hash_to_block`. **free only returns ownership; it does not clear the hash table.** This is also why preemption in the next section can "pick blocks back up".

The hit check has three layers, all on the kv_cache_manager side:

```
kv_cache_manager.get_computed_blocks         (kv_cache_manager.py)
  └── coordinator.find_longest_cache_hit     walk the block_hashes prefix chain, break at the first miss
        └── block_pool.get_cached_block      a dict lookup
```

Inside it there is this line:

```python
max_cache_hit_length = request.num_tokens - 1
```

Why cap the hit at `num_tokens − 1`? A full hit would mean this step computes zero tokens, so there are no logits to sample from; and the later `assert num_new_tokens > 0` would blow up. So even on a full hit, the last token is forced to be recomputed. The comment admits this may waste an entire block, because `num_computed_tokens` has to be block-aligned.

> Prefix cache hit = how many full blocks can be found along the block hash prefix chain. The hit count is written straight into num_computed_tokens.

## Preemption: gen3, who gets kicked, what is lost, how it comes back

gen3 sends both prompts: req 2 (PROMPT) hits 16 → computes 9, req 3 (PROMPT_B, 24 tokens) hits 0 → computes 24. After entering running, `len(self.running) = 2` and 5 blocks are free.

Both decode on their own. 9 blocks cannot hold two requests of length 128, given `max_model_len=128` and 10 blocks. So at some step, in the running section, req 3 needs a new block, `allocate_slots` returns None, and we enter the preemption loop at scheduler.py:424–468.

It is not:

```
foresee a future shortage
→ reject req 3 up front
```

But rather:

```
req 2 and req 3 both run first
→ KV grows dynamically with decode
→ memory actually runs out
→ one request fails to get a new block
→ kick one request out of running
→ free up blocks
→ the kicked one recomputes its way back later
```

### Who gets kicked

```python
preempted_req = self.running.pop()
self._preempt_request(preempted_req, scheduled_timestamp)
preempted_reqs.append(preempted_req)
if preempted_req == request:
    break
```

Under the FCFS policy it is simply `running.pop()`: **the tail of the queue, the most recent arrival**. Here, when free = 0, the request short of blocks is req 3, and the tail is also req 3. It kicks itself. `if preempted_req == request: break` exists for exactly this case: once you have been kicked yourself, there is nobody left to kick, and this step cannot schedule you.

I had assumed the preempted one is always "someone else". Seeing this line made it click: preemption exists to free blocks, and nothing says who gets to use them. This time the 4 freed blocks sat empty until req 2 finished.

### What is lost

`_preempt_request`, line by line:

```python
self.kv_cache_manager.free(request)      # free blocks 0 → 4 (64 tokens is exactly 4 blocks)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0          # 64 → 0
request.num_preemptions += 1
self.waiting.prepend_request(request)    # back to the head of waiting, not the tail
```

On entry `num_computed_tokens = 64` and `num_tokens = 64`; on exit `num_computed_tokens = 0`, but `num_tokens` is still 64. **What gets zeroed is "how far we have computed", not "how many tokens there are"**: the 24 prompt tokens and 40 output tokens all stay on the request object.

### How it comes back

On the next `schedule()`, req 3 is at the head of the waiting section.

```
status = PREEMPTED
num_tokens = 65
num_new_local_computed_tokens = 48
new_computed_blocks.get_block_ids() = ([3, 2, 6],)
```

The 3 full blocks just freed still have their hashes in the table, and the prefix cache picks them back up as they are. Step 2: `num_new_tokens = 65 − 48 = 17`. This is what that comment meant by "use `num_tokens` for resumed requests": when a preempted request returns, prompt tokens and already-generated output tokens are treated alike as "tokens that need to be computed up to".

Then Step 3 gets stuck: `allocate_slots` returns None, `break`. It needs 3 cached blocks + 2 new blocks = 5, and only 4 are free. It stalls here three times in a row, `get_num_free_blocks()` reading 4 each time, while req 3 lies at the head of waiting and req 2 keeps decoding.

When req 2 reaches 64 and finishes, its blocks are released, and the waiting section runs again. At station 3 this time `request.status == PREEMPTED`, so it goes into **`scheduled_resumed_reqs`**, `num_computed_tokens → 48`, and only 17 tokens are recomputed.

### Not the same as the paper's recompute

In the PagedAttention paper, recompute means drop everything and recompute everything: come back and redo all 65. In v1, `free()` only returns ownership and does not clear hashes, so a returning request first goes through the prefix cache and only recomputes the tail that never filled a block. **What is lost is ownership of the KV, not its content**: as long as a block has not been overwritten by someone else, it can be picked back up.

### A block that never got registered

After its final preemption, req 3 has 65 tokens. With `block_size=16`, the first 64 tokens should map to exactly 4 full blocks. Yet on resume, the prefix cache picked up only 3:

```
num_new_local_computed_tokens = 48
new_computed_blocks.get_block_ids() = ([3, 2, 6],)
```

Why did the 4th full block not come back?

At first I assumed it had been evicted. It had not.

After setting a breakpoint inside `cache_blocks()`, the real reason turned out to be:

> **The 4th block was already full and its hash already generated, but it had not yet been registered in the prefix cache when req 3 was preempted.**

The key distinction here:

```
the block is full
the block hash has been generated
the block has been registered in the prefix cache
```

These are three different moments.

#### Step one: 63 tokens, only 3 full blocks can be registered

One tick before the preemption, when schedule enters `allocate_slots()`:

```
num_tokens = 63
len(block_hashes) = 3
```

The token layout at this point:

```
[ 16 ][ 16 ][ 16 ][ 15 ]
```

So only the first three blocks are full. At the end, `allocate_slots()` executes:

```
cache_blocks(63)
```

The first three blocks are registered in the prefix cache.

#### Step two: token 64 comes back, the 4th block fills up, but is not registered

The GPU then returns an output token:

```
num_tokens: 63 → 64
len(block_hashes): 3 → 4
```

The layout is now:

```
[ 16 ][ 16 ][ 16 ][ 16 ]
```

So the 4th block's hash can now be computed.

But what the async scheduler calls at this moment is:

```python
self.kv_cache_manager.cache_blocks(
    request,
    request.num_computed_tokens - request.num_output_placeholders,
)
```

The state at this instant amounts to:

```
num_computed_tokens       = 64
num_output_placeholders   = 1
                           ----
registerable cache frontier = 63
```

So it still executes:

```
cache_blocks(63)
```

rather than `cache_blocks(64)`.

In other words, the 4th block's state right now is:

```
KV exists                       ✓
block hash generated            ✓
registered in prefix cache      ✗
```

This is the root of the "one block missing" later on.

#### Step three: the next tick should have registered the 4th block

Normally, when schedule enters `allocate_slots()` again on the next tick, it computes:

```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
```

which here would have given:

```
min(64 + 1, 64) = 64
```

and then reached:

```
cache_blocks(64)
```

at which point the 4th block would formally enter the prefix cache.

But this time it happens that:

```
free blocks = 0
```

`allocate_slots()` returns early at the capacity check that comes first:

```
the preemption tick
allocate_slots()
    ↓
num_blocks_to_allocate > free
    ↓
return None
```

So the `cache_blocks(64)` that should have run never happens.

The scheduler then immediately preempts req 3:

```
allocate_slots() → None
        ↓
_preempt_request(req 3)
        ↓
release the KV blocks held by req 3
        ↓
computed progress reset to zero
        ↓
req 3 goes back to waiting
```

So the 4th block is never registered in the prefix cache at any point.

#### Step four: why does it end up showing 65 tokens?

At the moment of preemption req 3 has 64 tokens, but one more computation has already been sent to the GPU and is in flight, and preemption does not cancel it. So after the preemption that output still comes back and still gets appended, which is why the debugger shows `status = PREEMPTED` and `num_tokens = 65`.

This 65 is a state that only appears after the preemption. It must not be plugged back into the earlier `64 − 1 = 63` call to `cache_blocks()`. Laying the four moments out in order:

```
① one tick before preemption
   num_tokens = 63
   → cache_blocks(63)
   → 3 blocks registered

② token 64 returns
   num_tokens = 64
   block_hashes = 4
   → async cache_blocks(64 - 1)
   → still only registered up to 63
   → 4th block has a hash but is not registered

③ the preemption tick
   should have been cache_blocks(64)
   but free = 0
   → allocate_slots() returns None early
   → req 3 is preempted

④ after preemption
   the earlier in-flight output returns
   num_tokens = 65
```

So on resume, the prefix cache can only find the three blocks that were actually registered:

```
new_computed_blocks.get_block_ids() = ([3, 2, 6],)
```

This is not eviction. The 4th block never went through:

```
registered → evicted
```

What it went through was:

```
filled up
→ hash generated
→ not yet registered
→ allocation fails on the next tick
→ preempt
→ registration opportunity missed
```

So the thing most worth remembering here:

> **Hash generation ≠ prefix cache registration. Async scheduling opens a brief gap between the two; this preemption happened to land inside that window, so a block that was already full and already hashed still could not be recovered by the prefix cache on resume.**

The whole preempt-and-resume process compresses to two lines:

```
preempt: free KV blocks + reset computed progress + back to waiting
resume:  query prefix cache → pick up registered blocks → recompute the rest
```

## Aside: has this section changed on main?

Last post mentioned that main turned the running section's `min()` into a three-argument `min(num_new_tokens, token_budget, input_budget − draft_slots)`. The waiting section is fully symmetric: the entry gains `if input_budget <= draft_slots: break`, the `min()` likewise takes three arguments, and after admission `input_budget -= num_new_tokens + draft_slots`. The two caps, `max_cache_hit_length = num_tokens − 1` and `num_tokens_to_cache`, are preserved as they are. The mechanism is unchanged; the budget just gained a dimension.

## Summary

- Admission = leftover budget from the running section + blocks available
- Prefix cache hit = how many full blocks are found along the block hash prefix chain; the hit count becomes num_computed_tokens directly
- Preemption = under FCFS, `running.pop()` kicks the newest, possibly itself; what gets zeroed is num_computed_tokens, not num_tokens
- Coming back = pick up the prefix cache first, then go through admission like a new request, landing in scheduled_resumed_reqs

---

At this point we have walked through both scheduling sections inside `schedule()`.

But what we have seen so far is only how the scheduler makes decisions: who keeps running, who gets blocks, who gets preempted when KV runs out. Once a tick has actually executed, the request's state keeps changing.

We already saw one example: req 3 still had 64 tokens when preempted, then an in-flight output returned and `num_tokens` became 65. But where exactly is that token written back to the request? Likewise: where is req 2 judged finished? And after a request ends, where are its KV blocks actually returned to the block pool?

None of this happens in the scheduling logic we have read so far. It all happens on the `update_from_output()` path, after GPU output returns. That is the next post.

</div>
