---
title: 'vLLM 源码追踪（二）· Scheduler 上：25 个 token 怎么变成 {16}→{9}→{1}'
title_en: 'Tracing vLLM Source (2) · Scheduler I: How 25 Tokens Become {16}→{9}→{1}'
date: 2026-09-03
permalink: /posts/2026/09/vllm-scheduler-1/
tags:
  - LLM
  - vLLM
  - 源码追踪
---

<div class="lang lang-zh" markdown="1">

本文基于 vLLM v0.21.0（V1 引擎，离线 `LLM.generate` 路径，关闭 multiprocessing，async scheduling 默认开启），用 debugger 追踪 `schedule()` 怎么把一条 25-token 的 prompt 切成三段 step。文中行号以该版本为准。

书接上回，这次我们把 `max_num_batched_tokens` 压到 16，一条 25-token 的 prompt 在调试器里变成了三段 step：`{16} → {9} → {1} {1} {1}…`

在开始今天的内容之前，我们得先看一下 `schedule` 的位置在哪里：

![debugger 停在 schedule 的调用栈](/images/posts/vllm-scheduler-1/fig1-callstack.png)

```
LLM.generate → _run_engine → LLMEngine.step → EngineCoreClient.get_output
→ EngineCore.step_with_batch_queue → Scheduler.schedule
```

v0.21 默认使用 async scheduling，因此 Core 侧的循环入口是 `step_with_batch_queue()`；它每轮调用 `scheduler.schedule()` 决定本轮要处理哪些 token。

问题一个一个来。

## Chunked prefill：step 1 和 step 2

### step 1

只 schedule 了 25 个里的 16 个。这就是 chunked prefill 切的第一刀，用完了 budget。

走的是 waiting 段里的 `min(num_new_tokens, token_budget)`，跟之后 running 段的表达式一样，只是位置不同。

```python
num_scheduled_tokens = {'0-xxxx': 16}
token_budget = 0
len(self.running) = 1
len(self.waiting) = 0
```

### step 2

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

25 + 0 − 16 = 9，`num_output_placeholders` 现在是 0。

`num_new_tokens = min(num_new_tokens, token_budget)`，即 min(9, 16) = 9——这里就是 chunked prefill。

step 1 之后 request 已经从 waiting 挪进 running（我们可以看到 `len(self.running) = 1`），scheduler 不管剩下的 9 个是 prompt 还是新 token。这里也能看出来 vLLM 不区分 prefill 和 decode：`schedule()` 里没有任何 `if is_prefill` 分支，每条 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec` 两个数。

`num_new_tokens = min(num_new_tokens, max_model_len - 1 - num_computed_tokens)`，还是 9，这是防 spec decode 越界的。

再往下进入 `allocate_slots` 循环：`token_budget` 剩 7、`new_blocks = ([2],)`。前 16 个 token 在 step 1 分了 block，现在为第 17–25 个再分一个。这里也是下篇要讲的 preempt 的入口：`allocate_slots` 返回 `None` 就会触发抢占。

剩下的 7 去哪了：留给 waiting 段给别的 request 准入。同一个 step 里这条 request 的 prefill 尾巴和别人的 decode 共用一个 budget。这就是 continuous batching。

**题外话**：到今天的最新版本，这一部分发生了变化，新增了一点东西：

```python
num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
```

两个 budget 的定义：

- `token_budget = max_num_scheduled_tokens`：这个 step 允许算多少 token。v0.21 里它直接由 `max_num_batched_tokens` 派生，所以就是我们压到的那个 16。
- `input_budget = max_num_batched_tokens`：model runner 输入缓冲区有多少个槽。每 schedule 一条 request 扣 `num_new_tokens + draft_slots`，不只扣 `num_new_tokens`。
  - `draft_slots`：并行 drafting 的 spec decode（P-EAGLE、DFlash、DSpark）每条 request 额外占的 slots；EAGLE3、MTP、ngram 这些不并行的为 0。

为什么拆：来自 commit `0914ed2e81`（2026-08-10，PR #51725，标题就是 "Adaptive budget for spec scheduled input tokens, ~60% better Kimi K3 DSpark TTFT"）。并行 draft 的槽不算 scheduled token，却实实在在占输入缓冲。所以拆成"算多少"和"槽够不够"两个约束，取更紧的那个。

> chunked prefill = 按 token_budget 把 prompt 切开，剩下的交给下一个 step

## Async scheduling：step 3 的那个 1 从哪来

### step 3

```python
num_computed_tokens = 25
num_tokens_with_spec = 25
num_output_placeholders = 1
num_new_tokens = 25 + 1 - 25 = 1
```

为什么"还差 0"却要 schedule 1 个？

因为 v0.21 默认 async scheduling：scheduler 在算 step 3 的时候，step 2 采样出来的那个 token 还没从 GPU 回来——它不知道 token id 是什么，但知道"一定会有一个"，所以先占一个位（placeholder）把 step 3 排上，GPU 不空转。

同步模式必须等：

```
GPU 生成 t26
↓
CPU 收到 t26
↓
Request 从 25 → 26
↓
Scheduler 才知道还差 1
```

async scheduling 类似于说：我虽然还不知道 t26 是 314 还是 9281，但我知道一定会生成一个 t26，所以提前记下我们还有一个 token 要计算。

Scheduler 找出 request 当前还没计算的 token，把它们安排进模型；模型执行后又产生新的 output token，下一轮继续处理。

> async scheduling = 用 placeholder 把下一 step 提前排上

---

到这里，`schedule()` 的 running 段我们走完了：chunked prefill 决定一条 request 这个 step 算多少，async scheduling 决定还没回来的 token 先占一个位。但 step 2 里 `allocate_slots` 那一步我们只是路过——如果它拿不到 block 会怎样？waiting 里的新请求又是怎么被放进来的？下篇讲 Scheduler 的准入与抢占。

</div>

<div class="lang lang-en" markdown="1">

This post is based on vLLM v0.21.0 (V1 engine, offline `LLM.generate` path, multiprocessing disabled, async scheduling on by default). Using a debugger, we trace how `schedule()` splits a 25-token prompt into three kinds of step. Line numbers refer to that version.

Picking up from last time: this time we squeeze `max_num_batched_tokens` down to 16, and a 25-token prompt turns into three kinds of step in the debugger: `{16} → {9} → {1} {1} {1}…`

Before we start, let's locate where `schedule` sits:

![Call stack paused at schedule](/images/posts/vllm-scheduler-1/fig1-callstack.png)

```
LLM.generate → _run_engine → LLMEngine.step → EngineCoreClient.get_output
→ EngineCore.step_with_batch_queue → Scheduler.schedule
```

v0.21 uses async scheduling by default, so the loop entry on the Core side is `step_with_batch_queue()`; every round it calls `scheduler.schedule()` to decide which tokens to process this round.

One question at a time.

## Chunked prefill: step 1 and step 2

### step 1

Only 16 of the 25 tokens get scheduled. This is chunked prefill's first cut — the budget is used up.

It goes through `min(num_new_tokens, token_budget)` in the waiting section, the same expression as in the running section later, just in a different place.

```python
num_scheduled_tokens = {'0-xxxx': 16}
token_budget = 0
len(self.running) = 1
len(self.waiting) = 0
```

### step 2

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

25 + 0 − 16 = 9; `num_output_placeholders` is 0 for now.

`num_new_tokens = min(num_new_tokens, token_budget)`, i.e. min(9, 16) = 9 — this is chunked prefill.

After step 1 the request has already moved from waiting into running (we can see `len(self.running) = 1`), and the scheduler doesn't care whether the remaining 9 are prompt tokens or new tokens. This also shows that vLLM does not distinguish prefill from decode: there is no `if is_prefill` branch anywhere in `schedule()`; each request is just two numbers, `num_computed_tokens` and `num_tokens_with_spec`.

`num_new_tokens = min(num_new_tokens, max_model_len - 1 - num_computed_tokens)` — still 9; this one guards against spec decode running past the model length.

Further down we enter the `allocate_slots` loop: `token_budget` has 7 left, `new_blocks = ([2],)`. The first 16 tokens got a block in step 1; now another block is allocated for tokens 17–25. This is also the entry point for preemption, which the next post covers: when `allocate_slots` returns `None`, preemption is triggered.

Where do the remaining 7 go? They are left for the waiting section, so other requests can be admitted. Within the same step, this request's prefill tail and someone else's decode share one budget. That is continuous batching.

**Aside**: as of the latest version, this part has changed slightly:

```python
num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
```

The two budgets:

- `token_budget = max_num_scheduled_tokens`: how many tokens this step is allowed to compute. In v0.21 it is derived directly from `max_num_batched_tokens`, so it is the 16 we squeezed it down to.
- `input_budget = max_num_batched_tokens`: how many slots the model runner's input buffer has. Each scheduled request deducts `num_new_tokens + draft_slots`, not just `num_new_tokens`.
  - `draft_slots`: extra slots per request used by parallel-drafting spec decode (P-EAGLE, DFlash, DSpark); non-parallel methods such as EAGLE3, MTP, and ngram use 0.

Why the split: it comes from commit `0914ed2e81` (2026-08-10, PR #51725, titled "Adaptive budget for spec scheduled input tokens, ~60% better Kimi K3 DSpark TTFT"). Parallel draft slots don't count as scheduled tokens, yet they really do occupy the input buffer. So the constraint is split into "how much to compute" and "are there enough slots", and the tighter one wins.

> chunked prefill = cut the prompt by token_budget and hand the rest to the next step

## Async scheduling: where step 3's 1 comes from

### step 3

```python
num_computed_tokens = 25
num_tokens_with_spec = 25
num_output_placeholders = 1
num_new_tokens = 25 + 1 - 25 = 1
```

Why schedule 1 token when "0 remain"?

Because v0.21 uses async scheduling by default: while the scheduler is computing step 3, the token sampled in step 2 hasn't come back from the GPU yet. It doesn't know the token id, but it knows "there will definitely be one", so it reserves a slot (a placeholder) and lines step 3 up right away, keeping the GPU from idling.

Synchronous mode has to wait:

```
GPU produces t26
↓
CPU receives t26
↓
Request goes from 25 → 26
↓
Only now does the Scheduler know 1 remains
```

Async scheduling is like saying: I don't yet know whether t26 is 314 or 9281, but I know a t26 will be generated, so I note ahead of time that we have one more token to compute.

The Scheduler finds each request's not-yet-computed tokens and lines them up for the model; after the model runs, new output tokens appear, and the next round continues.

> async scheduling = use a placeholder to line up the next step ahead of time

---

At this point we have walked through the running section of `schedule()`: chunked prefill decides how much of a request to compute in this step, and async scheduling reserves a slot for the token that hasn't come back yet. But we only passed by `allocate_slots` in step 2 — what happens when it can't get a block? And how do new requests in waiting get admitted? The next post covers the Scheduler's admission and preemption.

</div>
