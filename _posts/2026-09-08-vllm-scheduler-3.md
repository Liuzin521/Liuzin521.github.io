---
title: 'vLLM 源码追踪（四）· Scheduler 下：token 回来之后，记账、判停、还块'
title_en: 'Tracing vLLM Source (4) · Scheduler III: After the Token Returns, Bookkeeping, Stopping, Freeing'
date: 2026-09-08
permalink: /posts/2026/09/vllm-scheduler-3/
tags:
  - LLM
  - vLLM
  - 源码追踪
---

<div class="lang lang-zh" markdown="1">

本文基于 vLLM v0.21.0（V1 引擎，离线 `LLM.generate` 路径，关闭 multiprocessing，async scheduling 默认开启），用 debugger 追踪 `schedule()` 返回之后 scheduler 还做了什么：记账、收 token、判停、还块。文中行号以该版本为准。

书接上回。前两篇一直在 `schedule()` 里面看"这一步算多少"。这次看它外面，调试器里三个数对不上：

- step 1 刚 schedule 完，模型还没跑，`num_computed_tokens` 已经是 **25**
- 第 1 个 output token 到手时，`num_computed_tokens` 是 **26**，可 `num_tokens` 还是 25
- request 结束时，`num_tokens = 89`，`num_computed_tokens = 88`

配置和上篇一样：`max_model_len=128`、`max_num_seqs=2`、`num_gpu_blocks_override=10`，三次 generate（gen1 冷 / gen2 同 prompt / gen3 两条）。

## 位置：一轮循环里 scheduler 出场两次

上篇提到了一嘴 `EngineCore.step_with_batch_queue`。它一轮里和 scheduler 打两次交道（core.py）：

```
step_with_batch_queue
├── :475  scheduler.schedule()           → 决定这一步算什么，交给 GPU
└── :535  scheduler.update_from_output() → 拿回上一步的结果，记 token、判停、还块
```

前两篇在第一行里面。本篇的内容一半在 `schedule()` 尾巴上，一半在 `update_from_output()` 里。

问题一个一个来。

## 记账：模型没跑，25 是谁加的

`schedule()` 返回之前的最后一件事（scheduler.py:902）是调 `_update_after_schedule`：

```python
# scheduler.py:945
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
```

step 1 的 `num_scheduled_token = 25`，这一行一过 `num_computed_tokens` 就是 25。此刻 `execute_model` 还没被调用，GPU 上什么都没发生。

所以 **`num_computed_tokens` 的语义不是"模型算完了多少"，是"已经排上了多少"**。schedule 一返回账就记上，活是 GPU 之后干的。

我一开始以为这个数是模型跑完才加的。如果那样，下一轮 `schedule()` 就得等 GPU 回来才能算 `num_new_tokens`，上篇那个 placeholder 就没法存在。

> 记账在 schedule 时，不在 token 到手时。

## Placeholder：26 里那个 1 是什么

step 2 schedule 完，同一行让 `num_computed_tokens` 25 → 26，而 `num_output_tokens` 还是 0：step 1 的 output 还没回来。上篇讲过，那个 1 是 `num_output_placeholders`。

async 子类在 `_update_after_schedule` 之后紧跟着加它（async_scheduler.py:32）：

```python
request.num_output_placeholders += 1 + cur_num_spec_tokens
```

再往下，`update_from_output` 第一次拿到东西（scheduler.py:1308 `generated_token_ids`）：

```
generated_token_ids     = [576]
num_computed_tokens     = 26
num_output_tokens       = 0
num_output_placeholders = 2
```

第 1 个 step 的输出这时才到手，此时 step 2 已经排完。**placeholders = 2，就是这条 request in-flight 的 step 数。**

### 一条 request 怎么会有两个 step in-flight

这是我卡最久的地方：step 1 的 token 还没回来，step 2 的输入是什么？

答案分两半。**scheduler 不需要知道输入是哪个 token。** 它决定的是"几个"和"放哪"（`num_new_tokens = 1`、要不要新 block），整个 `schedule()` 里没有 token id。

**token id 留在 GPU 上，worker 自己填。** gpu_model_runner.py 两处：

```python
# :3497  step 1 采样完，结果留在 GPU，不同步回 CPU
self.input_batch.prev_sampled_token_ids = sampled_token_ids

# :1740  step 2 准备输入，直接在 GPU 上 scatter 进 input_ids
self.input_ids.gpu.scatter_(dim=0, index=..., src=self.input_batch.prev_sampled_token_ids[...])
```

"两个 in-flight"是对 CPU 说的。GPU 上 step 1 和 step 2 还是串行，step 2 开跑时 step 1 早已跑完，采出的 token 就在 `prev_sampled_token_ids` 里。placeholder 是 CPU 侧对"GPU 上已经有、我还没看见"的那个 token 的记账。

省掉的是 GPU → CPU → schedule → GPU 这一圈往返。同步模式下这一圈 GPU 在空转。

### 2 是哪来的

`step_with_batch_queue` 里有个队列（core.py:194）：

```python
self.batch_queue = deque(maxlen=self.batch_queue_size)
```

长度来自 uniproc_executor.py:78：

```python
return 2 if self.scheduler_config.async_scheduling else 1
```

每轮：队没满就 `schedule()` 一次入队；队满了就 `pop()` 最老的，`future.result()` 阻塞等它，交给 `update_from_output`。深度 2 → 最多 2 个 step 没回来 → 每条 request 的 placeholders 稳定在 2：每 schedule 一次 +1，每收回一个 output −1（async_scheduler.py:52）。

第 2 个 output 到手时的数印证了这个节奏：`[12884]`，`num_computed_tokens = 27`，`num_tokens = 26`。

> placeholders = 这条 request in-flight 的 step 数 = batch queue 深度。

## 结束：88 和 89 差的是谁

每个 output token 回来都走 `_update_request_with_output`（scheduler.py:1559）：

```python
request.append_output_token_ids(new_token_ids)     # num_tokens += 1
stopped = check_stop(request, self.max_model_len)  # :1571
```

`check_stop`（utils.py:94）按这个顺序判：

1. `num_output_tokens < min_tokens` → `return False`（还不许停；默认 0，跳过）
2. 最后一个 token == `eos_token_id` → FINISHED_STOPPED
3. 最后一个 token in `stop_token_ids`（用户自定义的额外停止符，我们没设）→ FINISHED_STOPPED
4. `num_tokens >= max_model_len` 或 `num_output_tokens >= max_tokens` → FINISHED_LENGTH_CAPPED

gen1 的第 64 个 output 回来时命中第 4 条：

```
status = FINISHED_LENGTH_CAPPED
num_output_tokens = 64,  max_tokens = 64
num_tokens = 89,  num_computed_tokens = 88
get_finished_reason() = LENGTH
```

之后 `_free_request` 还块，从 running 摘掉，finish_reason=length 装进 EngineCoreOutput 回前端。

**88 和 89 差的那个，不是 placeholder。** 是第 64 个 output 自己。decode 每一步是"拿上一个 output 当输入，产出下一个"，所以 output k 的 KV 在产出 output k+1 那步才算。output 64 之后没有下一步了，它没当过输入，没过 transformer，没有 KV。

**`num_computed_tokens` 数的是当过模型输入的 token，最后一个 output 永远不在里面。**

### 一个 guard

结束那一刻 `num_output_placeholders = 1`，不是 2。按 batch queue 深 2 的节奏，output 64 到手时 step 65 应该已经排上了。没排，是 `schedule()` running 段开头有个 guard（scheduler.py:350–364）：

```python
if (request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens):
    req_index += 1
    continue      # 确定上一步已到 max_tokens，不再多排一步
```

所以"async 下 request 结束会白算一个 token"这个直觉要收窄：**只有 EOS 和 stop token 会**——schedule 时不知道会采出它们。max_tokens 是事先知道的，vLLM 提前躲掉了。

> 结束 = append → check_stop 四层 → FINISHED → 还块 → 摘出 running。

## 还块：free 了，但没丢

request 结束后 `_free_request → _free_blocks`（scheduler.py:1768）：

```python
self.kv_cache_manager.free(request)
```

gen2 结束时这一行前后的数：

```
get_block_ids()                       ([1, 7, 8, 9, 6, 5],)
get_num_free_blocks()                 3  →  9
len(cached_block_hash_to_block)       5  →  5
```

free 3 → 9，hash 表条数 **5 → 5**。`free()` 只把 6 个 block 放回 free 队列，hash 一条没删。上篇 gen2 能命中 block 1、req 3 被抢后能捡回 48，靠的就是这个。

两个顺手看到的细节：

- **block_ids 为什么是 `[1,7,8,9,6,5]` 不是 `[1..6]`**：block 1 是 prefix cache 命中复用的，其余 5 块从 free 队列按顺序拿。gen1 结束时 6 块倒序放回队尾，队列是 `7 8 9 | 6 5 4 3 2 1`，没用过的在前。2、3、4 没人碰，hash 还挂着。没有任何"替换"发生。
- **hash 条数为什么还是 5**：同 prompt + greedy，64 个 output 一模一样，每块 hash 一模一样，插进去等于没插。

gen3 把上篇的抢占收了尾。req 2 先结束，`free` 之前：

```
get_block_ids()                       ([1, 4, 5, 9, 7, 8],)
len(cached_block_hash_to_block)       8        # PROMPT 链 5 + req 3 留下的 [3,2,6]
self.waiting.peek_request().status    PREEMPTED
```

waiting 队头躺着上篇被抢的 req 3。req 2 这 6 块一还，下一次 `schedule()` 它就带着 48 个命中以 `scheduled_resumed_reqs` 回来。req 3 后结束时 `block_ids = ([3, 2, 6, 8, 7, 9],)`：前 3 块是捡回来的，后 3 块是 req 2 刚还的。`num_tokens / num_computed_tokens = 88 / 87`，又是差 1。

> free = 还所有权，hash 表不动；下一个同前缀的 request 直接捡。

## 题外话：main 上这一段变了吗

`check_stop` 的顺序变了：main 把 `min_tokens` 从第 1 条挪到了长度判断之后（EOS → stop_token_ids → 长度 → min_tokens）。v0.21.0 的顺序下，`min_tokens > max_tokens` 时长度判断会被 min_tokens 的 `return False` 挡住，request 停不下来。

async 这边：`placeholders += 1 + spec` 泛化成 `num_sampled_tokens_per_step + spec`；被抢 request 的 in-flight 输出从 `discard_latest_async_tokens` 换成了 `num_stale_output_tokens` 机制，stale 的输出不再减 placeholders。记账、判停、还块的骨架没变。

## 小结

- 记账在 `schedule()` 返回时（`_update_after_schedule`），token 在 `update_from_output` 到手，中间差的是 placeholder：

```
schedule()
↓
决定：我要算下一 token

_update_after_schedule()
↓
先记：有 1 个 output 在路上
placeholder +1

    【GPU 在算】

update_from_output()
↓
真实 output token 回来
num_tokens +1
placeholder -1
```

- placeholders = 这条 request in-flight 的 step 数 = batch queue 深度 2；token id 留在 GPU，scheduler 只管几个、放哪
- 结束 = append → check_stop（min_tokens → EOS → stop_token_ids → 长度）→ FINISHED → 还块 → 摘出 running；max_tokens 有 guard 不白算，EOS 白算 1 个
- `num_computed_tokens` 数的是当过输入的 token，最后一个 output 不在里面，所以永远差 1
- free 只还所有权，hash 表不删；被抢的、同前缀的下次直接捡

---

到这里 Scheduler 三篇走完：上篇 running 段决定每条 request 算多少，中篇 waiting 段决定谁能进来、谁被踢，本篇 `update_from_output` 收账。三篇里 `allocate_slots` 一直是个黑盒：它凭什么说"够"或"不够"，block 和 hash 是怎么对上的，`[1,7,8,9,6,5]` 这个顺序谁定的。接下来我们会进 kv_cache_manager。

</div>

<div class="lang lang-en" markdown="1">

This post is based on vLLM v0.21.0 (V1 engine, offline `LLM.generate` path, multiprocessing disabled, async scheduling on by default). Using a debugger, we trace what the scheduler still does after `schedule()` returns: bookkeeping, collecting tokens, deciding when to stop, and returning blocks. Line numbers refer to that version.

Picking up from last time. The previous two posts stayed inside `schedule()`, watching "how much to compute this step". This time we look outside it, where three numbers in the debugger do not add up:

- Right after step 1 is scheduled, before the model has run, `num_computed_tokens` is already **25**
- When the 1st output token arrives, `num_computed_tokens` is **26**, yet `num_tokens` is still 25
- When the request finishes, `num_tokens = 89` and `num_computed_tokens = 88`

Same configuration as last post: `max_model_len=128`, `max_num_seqs=2`, `num_gpu_blocks_override=10`, three generate calls (gen1 cold / gen2 same prompt / gen3 two prompts).

## Where: the scheduler appears twice per loop iteration

Last post briefly mentioned `EngineCore.step_with_batch_queue`. In one iteration it talks to the scheduler twice (core.py):

```
step_with_batch_queue
├── :475  scheduler.schedule()           → decide what this step computes, hand it to the GPU
└── :535  scheduler.update_from_output() → take back the previous step's result, record tokens, check stop, free blocks
```

The previous two posts lived inside the first line. This post is half at the tail of `schedule()` and half inside `update_from_output()`.

One question at a time.

## Bookkeeping: the model has not run, so who added 25

The last thing `schedule()` does before returning (scheduler.py:902) is call `_update_after_schedule`:

```python
# scheduler.py:945
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
```

For step 1, `num_scheduled_token = 25`, and as soon as this line runs `num_computed_tokens` is 25. At this moment `execute_model` has not been called; nothing has happened on the GPU.

So **the meaning of `num_computed_tokens` is not "how much the model has finished computing" but "how much has been scheduled"**. The books are updated the moment schedule returns; the GPU does the work afterwards.

I had assumed this number was incremented after the model ran. If it were, the next `schedule()` would have to wait for the GPU to come back before it could compute `num_new_tokens`, and the placeholder from last post could not exist.

> Bookkeeping happens at schedule time, not when the token arrives.

## Placeholder: what is the 1 in 26

After step 2 is scheduled, the same line takes `num_computed_tokens` from 25 → 26, while `num_output_tokens` is still 0: step 1's output has not come back. As covered last post, that 1 is `num_output_placeholders`.

The async subclass adds it right after `_update_after_schedule` (async_scheduler.py:32):

```python
request.num_output_placeholders += 1 + cur_num_spec_tokens
```

Further down, the first time `update_from_output` gets something (scheduler.py:1308, `generated_token_ids`):

```
generated_token_ids     = [576]
num_computed_tokens     = 26
num_output_tokens       = 0
num_output_placeholders = 2
```

Step 1's output only arrives now, and by this point step 2 has already been scheduled. **placeholders = 2 is the number of steps this request has in flight.**

### How can one request have two steps in flight

This is where I was stuck longest: step 1's token has not come back, so what is step 2's input?

The answer has two halves. **The scheduler does not need to know which token the input is.** It decides "how many" and "where" (`num_new_tokens = 1`, whether a new block is needed). There is no token id anywhere in `schedule()`.

**The token id stays on the GPU, and the worker fills it in itself.** Two places in gpu_model_runner.py:

```python
# :3497  after step 1 samples, the result stays on the GPU, no sync back to CPU
self.input_batch.prev_sampled_token_ids = sampled_token_ids

# :1740  preparing step 2's input, scatter straight into input_ids on the GPU
self.input_ids.gpu.scatter_(dim=0, index=..., src=self.input_batch.prev_sampled_token_ids[...])
```

"Two in flight" is from the CPU's point of view. On the GPU, step 1 and step 2 are still serial: by the time step 2 starts, step 1 has long finished and the sampled token is sitting in `prev_sampled_token_ids`. The placeholder is the CPU side's bookkeeping for a token that "already exists on the GPU but I have not seen yet".

What gets saved is the GPU → CPU → schedule → GPU round trip. In synchronous mode the GPU idles for that whole loop.

### Where the 2 comes from

`step_with_batch_queue` has a queue (core.py:194):

```python
self.batch_queue = deque(maxlen=self.batch_queue_size)
```

Its length comes from uniproc_executor.py:78:

```python
return 2 if self.scheduler_config.async_scheduling else 1
```

Each iteration: if the queue is not full, call `schedule()` once and enqueue; if it is full, `pop()` the oldest, block on its `future.result()`, and hand it to `update_from_output`. Depth 2 → at most 2 steps outstanding → each request's placeholders settle at 2: +1 per schedule, −1 per output received (async_scheduler.py:52).

The numbers when the 2nd output arrives confirm the rhythm: `[12884]`, `num_computed_tokens = 27`, `num_tokens = 26`.

> placeholders = the number of steps this request has in flight = batch queue depth.

## Finishing: what is the difference between 88 and 89

Every returning output token goes through `_update_request_with_output` (scheduler.py:1559):

```python
request.append_output_token_ids(new_token_ids)     # num_tokens += 1
stopped = check_stop(request, self.max_model_len)  # :1571
```

`check_stop` (utils.py:94) checks in this order:

1. `num_output_tokens < min_tokens` → `return False` (not allowed to stop yet; default 0, skipped)
2. Last token == `eos_token_id` → FINISHED_STOPPED
3. Last token in `stop_token_ids` (user-defined extra stop tokens; we set none) → FINISHED_STOPPED
4. `num_tokens >= max_model_len` or `num_output_tokens >= max_tokens` → FINISHED_LENGTH_CAPPED

When gen1's 64th output arrives it hits rule 4:

```
status = FINISHED_LENGTH_CAPPED
num_output_tokens = 64,  max_tokens = 64
num_tokens = 89,  num_computed_tokens = 88
get_finished_reason() = LENGTH
```

After that, `_free_request` returns the blocks, the request is removed from running, and finish_reason=length is packed into an EngineCoreOutput back to the frontend.

**The one that separates 88 from 89 is not a placeholder.** It is the 64th output itself. Every decode step is "take the previous output as input, produce the next one", so output k's KV is computed in the step that produces output k+1. After output 64 there is no next step: it never served as input, never went through the transformer, has no KV.

**`num_computed_tokens` counts tokens that have been model input, and the last output is never among them.**

### A guard

At the moment of finishing, `num_output_placeholders = 1`, not 2. At the batch queue's depth-2 rhythm, step 65 should already have been scheduled by the time output 64 arrives. It was not, because of a guard at the start of the running section in `schedule()` (scheduler.py:350–364):

```python
if (request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens):
    req_index += 1
    continue      # sure the previous step reached max_tokens; do not schedule one more
```

So the intuition that "under async a request wastes one token's compute when it finishes" needs narrowing: **only EOS and stop tokens do**, because at schedule time nobody knows they will be sampled. max_tokens is known in advance, and vLLM sidesteps it.

> Finishing = append → four check_stop rules → FINISHED → free blocks → remove from running.

## Freeing: freed, but not lost

After a request finishes, `_free_request → _free_blocks` (scheduler.py:1768):

```python
self.kv_cache_manager.free(request)
```

The numbers before and after this line when gen2 finishes:

```
get_block_ids()                       ([1, 7, 8, 9, 6, 5],)
get_num_free_blocks()                 3  →  9
len(cached_block_hash_to_block)       5  →  5
```

Free goes 3 → 9, and the hash table entry count is **5 → 5**. `free()` only puts the 6 blocks back on the free queue; not one hash is deleted. This is what let gen2 hit block 1 last post, and what let req 3 pick up 48 tokens after being preempted.

Two details noticed along the way:

- **Why block_ids is `[1,7,8,9,6,5]` and not `[1..6]`**: block 1 is reused from a prefix cache hit; the other 5 are taken from the free queue in order. When gen1 finished, its 6 blocks were pushed back to the tail in reverse, so the queue was `7 8 9 | 6 5 4 3 2 1`, never-used blocks first. Blocks 2, 3, 4 are untouched and their hashes still hang there. No "replacement" happened at all.
- **Why the hash count is still 5**: same prompt + greedy, so all 64 outputs are identical, every block hash is identical, and inserting is a no-op.

gen3 closes out last post's preemption. req 2 finishes first; just before `free`:

```
get_block_ids()                       ([1, 4, 5, 9, 7, 8],)
len(cached_block_hash_to_block)       8        # 5 from the PROMPT chain + [3,2,6] left by req 3
self.waiting.peek_request().status    PREEMPTED
```

At the head of waiting lies req 3, preempted last post. As soon as req 2 returns these 6 blocks, the next `schedule()` brings req 3 back via `scheduled_resumed_reqs` with 48 cache hits. When req 3 finishes later, `block_ids = ([3, 2, 6, 8, 7, 9],)`: the first 3 are the ones picked back up, the last 3 are the ones req 2 just returned. `num_tokens / num_computed_tokens = 88 / 87`, off by one again.

> free = return ownership, leave the hash table alone; the next request with the same prefix picks it up directly.

## Aside: has this section changed on main?

The order in `check_stop` changed: main moved `min_tokens` from rule 1 to after the length check (EOS → stop_token_ids → length → min_tokens). Under v0.21.0's order, when `min_tokens > max_tokens` the length check is blocked by min_tokens' `return False`, and the request cannot stop.

On the async side: `placeholders += 1 + spec` is generalized to `num_sampled_tokens_per_step + spec`; the in-flight output of a preempted request moved from `discard_latest_async_tokens` to a `num_stale_output_tokens` mechanism, and stale outputs no longer decrement placeholders. The skeleton of bookkeeping, stopping and freeing is unchanged.

## Summary

- Bookkeeping happens when `schedule()` returns (`_update_after_schedule`); the token arrives in `update_from_output`; the gap in between is the placeholder:

```
schedule()
↓
decide: I will compute the next token

_update_after_schedule()
↓
record first: 1 output is on its way
placeholder +1

    [GPU computing]

update_from_output()
↓
the real output token comes back
num_tokens +1
placeholder -1
```

- placeholders = the number of steps this request has in flight = batch queue depth 2; token ids stay on the GPU, the scheduler only handles how many and where
- Finishing = append → check_stop (min_tokens → EOS → stop_token_ids → length) → FINISHED → free blocks → remove from running; max_tokens has a guard so nothing is wasted, EOS wastes 1
- `num_computed_tokens` counts tokens that have been input; the last output is not among them, hence always off by one
- free only returns ownership and does not delete hashes; preempted requests and same-prefix requests pick them up next time

---

That completes the three Scheduler posts: part I, the running section deciding how much each request computes; part II, the waiting section deciding who gets in and who gets kicked; this part, `update_from_output` settling the books. Across all three, `allocate_slots` has stayed a black box: on what basis does it say "enough" or "not enough", how do blocks and hashes line up, and who decided the order `[1,7,8,9,6,5]`. Next we go into kv_cache_manager.

</div>
