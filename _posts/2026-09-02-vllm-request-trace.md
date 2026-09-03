---
title: 'vLLM 源码追踪（一）· 一条请求的四次变身：从 LLM.generate() 到 Scheduler'
date: 2026-09-02
permalink: /posts/2026/09/vllm-request-trace/
tags:
  - LLM
  - vLLM
  - 源码追踪
---

<div class="lang lang-zh" markdown="1">

本文基于 vLLM v0.21.0（V1 引擎，离线 `LLM.generate` 路径，关闭 multiprocessing），用 debugger 追踪一条请求从 `generate()` 到进入 Scheduler waiting 队列的全过程。文中行号以该版本为准。

下面是我们今天用的 Prompt：

```python
PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Explain, step by step, why the sky appears blue during the day."
)
```

先留一个 vLLM 中代码对象的整体初印象：

```
LLM
 ↓
LLMEngine
 ↓
EngineCoreClient
 ↓
EngineCore
 ↓
Executor
 ↓
GPUWorker / ModelRunner
```

![vLLM 请求生命周期总览](/images/posts/vllm-request-trace/fig1-overview.png)

## request 的四次变身

### 变身 1：str → EngineInput（翻译，tokenizer）· 前端 renderer

![debugger 停在 _tokenize_prompt 的调用栈](/images/posts/vllm-request-trace/fig2-tokenize.png)

- **在哪**：`renderers/base.py:407` `_tokenize_prompt` → `tokenizer.encode`
- **出来长什么样**：文字和 25 个 id 并存的字典

```python
prompt_token_ids = [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13,
                    81917, 11, 3019, 553, 3019, 11, 3170, 279, 12884, 7952,
                    6303, 2337, 279, 1899, 13]
prompt = {'prompt': 'The quick brown fox jumps over the lazy dog. Explain, step by step, why the sky appears blue during the day.'}
```

- 栈里还没有 `llm_engine`，`add_request` 还没调用。

### 变身 2：EngineInput → EngineCoreRequest（装袋）· llm_engine.add_request

- **在哪**：`llm_engine.py:240` `input_processor.process_inputs`
- **三件事**：装袋 / `assign_request_id` 把 `'0'` 改成 `'0-8f77ff70'`（8 位随机后缀保证唯一，原 id 存进 `external_req_id`）/ `:252` 抽出 `prompt_text` → `:265` 交给 `output_processor.add_request` 留在前端

```python
prompt = {'type': 'token', 'prompt_token_ids': [785, 3974, 13876, ...],
          'prompt': 'The quick brown fox jumps over the lazy dog. ...',
          'arrival_time': 1788369226.667193}

# 装好的 EngineCoreRequest
request = EngineCoreRequest(request_id='0-8f77ff70',
                            prompt_token_ids=[785, 3974, 13876, 38835, 34208, 916, 279,
                                              15678, 5562, 13, 81917, 11, 3019, 553, 3019,
                                              11, 3170, 279, 12884, 7952, 6303, 2337, 279, 1899, 13],
                            external_req_id='0', ...)
```

之前在 `prompt` 里的英文原文被抽取出来放到了 `prompt_text`，交给前端的 `output_processor` 留档；之后 `self.engine_core.add_request(request)`，只带数字过桥。

- **为什么文字留前端**：detokenize 是前端的活（拼流式输出要上下文）；核心的热循环里不该有任何字符串操作。

### 变身 3：EngineCoreRequest → Request（建档）· 过桥进核心

- **在哪**：`core_client.py:296-297` → `core.py:783` `Request.from_engine_core_request`

![debugger 中新建的 Request 对象](/images/posts/vllm-request-trace/fig3-request.png)

- **加了什么**：`status = WAITING`（`request.py:97`）、`num_computed_tokens = 0`（`:146`）

新建的 `req` 跟之前的 `request`（EngineCoreRequest）对比，多出了调度用的字段：

- `req.status` → `WAITING`
- `req.num_computed_tokens` → `0`
- `req.prompt_token_ids` 还是那 25 个

建档 = 加调度状态 + 提前算好 block hash。

- **意外发现**：`Request.__init__` 末尾就调 `update_block_hashes()`（`request.py:233`）——prefix caching 的 block hash 在建档时就算好了。25 个 token = 1 个满 block（16）→ 1 个 hash，`len(req.block_hashes) = 1`。
- **为什么要三种 request 类型**：API 层好用的 request / CoreClient 用来传递的 request / Scheduler 真正管理的 request。

### 变身 4：Request → waiting 队列（排队）· Scheduler.add_request

![debugger 中 waiting 队列的变化](/images/posts/vllm-request-trace/fig4-waiting.png)

- `scheduler.py:1681`：`_enqueue_waiting_request(request)` + `self.requests[request_id] = request`
- 调用链：`scheduler.add_request` ← `EngineCore.add_request` ← `core_client.add_request` ← `llm_engine.add_request` ← … ← `generate`
- `len(self.waiting)` 和 `len(self.requests)` 一开始都是 0。然后 `existing = self.requests.get(...)` 拿到 `None`，进 else 分支，跨过 `self._enqueue_waiting_request(request)` 和 `self.requests[request.request_id] = request` 这两行，两个长度变成 1 和 1。这条请求从此躺在 waiting 里，等下一个 step `schedule()` 来。

## 自测 3 问

**Q1：tokenize 发生在 vLLM 的哪一层？引擎核心见过字符串吗？**

在 vLLM 前端 LLM 部分，EngineCore 自始至终没见过字符串。组件是 renderer，函数是 `renderers/base.py` 的 `_tokenize_prompt`，里面调 `tokenizer.encode`。

**Q2：一条请求从用户到调度器换了几种对象？各自为什么存在？**

换了 3 个对象（前面说"四次变身"，但第四次是进 waiting 队列——对象没变，变的是它的处境）：API 层好用的 request / CoreClient 用来传递的 request / Scheduler 真正管理的 request。

为什么不直接把 `Request` 跨进程传过去？

- 它是 Core 内部的运行时状态对象——`status`、`num_computed_tokens`、`block_hashes` 等字段会随着调度不断变化，应该由 EngineCore / Scheduler 单独维护；如果前端也拿着同一个语义上的 `Request`，就会引入状态同步和所有权问题。
- 跨进程边界应该传稳定的"纯数据"，而不是 Core 内部对象——`EngineCoreRequest` 只携带 Core 创建请求所需的数据，适合作为 IPC 的传输格式；到了 Core 之后，再根据这些数据构造带有调度状态和派生字段的 `Request`。

**Q3：为什么 V1 需要 EngineCoreClient？**

LLMEngine 不直接关心 EngineCore 在不在同一个进程。两者同进程时直接调用；分属不同进程时，通过 ZMQ 传消息。EngineCoreClient 把这两种通信方式统一成同一个接口。

同进程 `InprocClient`，多进程 `SyncMPClient`（离线）或 `AsyncMPClient`（在线 vllm serve）。`make_client` 按两个开关选：`multiprocess_mode`（就是 `VLLM_ENABLE_V1_MULTIPROCESSING`）和 `asyncio_mode`。我们脚本关了 multiprocess，所以走 `InprocClient`，client 直接调 `engine_core.add_request`。

---

到这里，请求只是躺进了 waiting 队列。下一个 step，`schedule()` 会怎么决定谁先跑、给它分多少 KV cache block？下篇接着 trace。

</div>

<div class="lang lang-en" markdown="1">

This post is based on vLLM v0.21.0 (V1 engine, offline `LLM.generate` path, multiprocessing disabled). Using a debugger, we trace one request all the way from `generate()` until it enters the Scheduler's waiting queue. Line numbers refer to that version.

Here is the prompt we use today:

```python
PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Explain, step by step, why the sky appears blue during the day."
)
```

First, a rough mental map of the code objects in vLLM:

```
LLM
 ↓
LLMEngine
 ↓
EngineCoreClient
 ↓
EngineCore
 ↓
Executor
 ↓
GPUWorker / ModelRunner
```

![vLLM request lifecycle overview](/images/posts/vllm-request-trace/fig1-overview.png)

## The request's four transformations

### Transformation 1: str → EngineInput (translation, tokenizer) · frontend renderer

![Call stack paused at _tokenize_prompt](/images/posts/vllm-request-trace/fig2-tokenize.png)

- **Where**: `renderers/base.py:407` `_tokenize_prompt` → `tokenizer.encode`
- **What comes out**: a dict holding both the text and 25 token ids

```python
prompt_token_ids = [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13,
                    81917, 11, 3019, 553, 3019, 11, 3170, 279, 12884, 7952,
                    6303, 2337, 279, 1899, 13]
prompt = {'prompt': 'The quick brown fox jumps over the lazy dog. Explain, step by step, why the sky appears blue during the day.'}
```

- `llm_engine` is not on the stack yet; `add_request` has not been called.

### Transformation 2: EngineInput → EngineCoreRequest (bagging) · llm_engine.add_request

- **Where**: `llm_engine.py:240` `input_processor.process_inputs`
- **Three things happen**: bagging / `assign_request_id` turns `'0'` into `'0-8f77ff70'` (an 8-char random suffix guarantees uniqueness; the original id is kept in `external_req_id`) / line `:252` extracts `prompt_text` → line `:265` hands it to `output_processor.add_request`, which stays in the frontend

```python
prompt = {'type': 'token', 'prompt_token_ids': [785, 3974, 13876, ...],
          'prompt': 'The quick brown fox jumps over the lazy dog. ...',
          'arrival_time': 1788369226.667193}

# the packed EngineCoreRequest
request = EngineCoreRequest(request_id='0-8f77ff70',
                            prompt_token_ids=[785, 3974, 13876, 38835, 34208, 916, 279,
                                              15678, 5562, 13, 81917, 11, 3019, 553, 3019,
                                              11, 3170, 279, 12884, 7952, 6303, 2337, 279, 1899, 13],
                            external_req_id='0', ...)
```

The English text that used to live in `prompt` has been extracted into `prompt_text` and archived with the frontend's `output_processor`; after that, `self.engine_core.add_request(request)` crosses the bridge carrying numbers only.

- **Why the text stays in the frontend**: detokenization is the frontend's job (assembling streamed output needs context); the core's hot loop should never touch a string.

### Transformation 3: EngineCoreRequest → Request (filing) · crossing into the core

- **Where**: `core_client.py:296-297` → `core.py:783` `Request.from_engine_core_request`

![The newly built Request object in the debugger](/images/posts/vllm-request-trace/fig3-request.png)

- **What gets added**: `status = WAITING` (`request.py:97`), `num_computed_tokens = 0` (`:146`)

Compared with the earlier `request` (EngineCoreRequest), the new `req` gains scheduling fields:

- `req.status` → `WAITING`
- `req.num_computed_tokens` → `0`
- `req.prompt_token_ids` — still those same 25

Filing = adding scheduling state + precomputing block hashes.

- **A surprise**: the end of `Request.__init__` already calls `update_block_hashes()` (`request.py:233`) — the block hashes for prefix caching are computed at filing time. 25 tokens = 1 full block (of 16) → 1 hash, so `len(req.block_hashes) = 1`.
- **Why three request types**: a request that's convenient at the API layer / a request the CoreClient uses for transport / a request the Scheduler actually manages.

### Transformation 4: Request → waiting queue (queuing) · Scheduler.add_request

![The waiting queue changing in the debugger](/images/posts/vllm-request-trace/fig4-waiting.png)

- `scheduler.py:1681`: `_enqueue_waiting_request(request)` + `self.requests[request_id] = request`
- Call chain: `scheduler.add_request` ← `EngineCore.add_request` ← `core_client.add_request` ← `llm_engine.add_request` ← … ← `generate`
- `len(self.waiting)` and `len(self.requests)` both start at 0. Then `existing = self.requests.get(...)` returns `None`, we take the else branch, step over `self._enqueue_waiting_request(request)` and `self.requests[request.request_id] = request`, and both lengths become 1. From now on this request lies in waiting, until the next step's `schedule()` comes for it.

## Three self-test questions

**Q1: At which layer of vLLM does tokenization happen? Does the engine core ever see a string?**

In the vLLM frontend, the LLM part. The EngineCore never sees a string, start to finish. The component is the renderer; the function is `_tokenize_prompt` in `renderers/base.py`, which calls `tokenizer.encode`.

**Q2: How many object types does one request pass through from the user to the scheduler, and why does each exist?**

Three objects (we said "four transformations", but the fourth is entering the waiting queue — the object doesn't change, its situation does): a request that's convenient at the API layer / a request the CoreClient uses for transport / a request the Scheduler actually manages.

Why not just send `Request` across the process boundary?

- It is the Core's internal runtime-state object — fields like `status`, `num_computed_tokens`, and `block_hashes` keep changing as scheduling proceeds, and should be maintained solely by EngineCore / Scheduler. If the frontend also held the semantically same `Request`, you'd get state-synchronization and ownership problems.
- What crosses a process boundary should be stable "pure data", not a Core-internal object — `EngineCoreRequest` carries only the data the Core needs to create a request, which makes it a good IPC transport format. Once inside the Core, a `Request` with scheduling state and derived fields is constructed from that data.

**Q3: Why does V1 need an EngineCoreClient?**

LLMEngine doesn't directly care whether EngineCore lives in the same process. In the same process, it's a direct call; in different processes, messages go over ZMQ. EngineCoreClient unifies those two communication styles behind one interface.

Same process → `InprocClient`; multiprocess → `SyncMPClient` (offline) or `AsyncMPClient` (online, vllm serve). `make_client` chooses by two switches: `multiprocess_mode` (i.e. `VLLM_ENABLE_V1_MULTIPROCESSING`) and `asyncio_mode`. Our script disables multiprocessing, so we get `InprocClient`, and the client calls `engine_core.add_request` directly.

---

At this point, the request is merely lying in the waiting queue. In the next step, how does `schedule()` decide who runs first, and how many KV cache blocks to give it? We'll keep tracing in the next post.

</div>
