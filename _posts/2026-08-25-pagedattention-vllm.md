---
title: '从 PagedAttention 原文走进 vLLM'
title_en: 'From the PagedAttention Paper into vLLM'
date: 2026-08-25
permalink: /posts/2026/08/pagedattention-vllm/
tags:
  - LLM
  - vLLM
  - 论文笔记
---

<div class="lang lang-zh" markdown="1">

![PagedAttention 论文](/images/posts/pagedattention/fig1-cover.png)

2023 年的 vLLM 论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》大概是 LLM serving 领域被引用最多的工作之一。今天几乎所有推理框架都或多或少继承了它的思想。它回答的问题其实很朴素：**GPU 显存里的 KV cache 应该怎么管？** 论文给出的答案是把操作系统虚拟内存的分页思想搬过来：KV cache 不再要求连续存储，而是切成固定大小的 block，用一张 block table 做逻辑到物理的映射。就这一个改动，把内存利用率从不到 40% 提到了 96% 以上，吞吐量翻了 2-4 倍。

这篇文章是我读原文的个人理解，按论文的脉络走三步：**为什么 KV cache 的管理直接决定吞吐（动机）→ PagedAttention 怎么设计（分页、共享、调度）→ vLLM 怎么落地（kernel 优化）**。

## 动机：KV cache 的管理 = 吞吐

在服务的时候，如果能多个请求同时服务（也就是增大 batch 的数量），是显然能够增加系统的吞吐量的。

![显存占用分布](/images/posts/pagedattention/fig2-memory.png)

用 A100 40G 跑 13B 的模型，KV cache 占约 30%。所以能同时服务多少请求这个问题，就能转移到 KV cache 是怎么管的上面。

因为自回归 LLM 的 decode 部分是逐 token 生成的，每一步的生成仅仅是 matrix-vector 乘法，会导致 GPU 的算力利用不足，所以提升吞吐的方法就是加大 batch 的数量，让一次权重的搬运开销摊薄到多个 batch 上。但是 batch size 也不能随意增加，这里的上限就取决于 KV cache 的管理——**内存效率就是吞吐**，也说明了这一部分是 memory bound。

与此同时，prefill 部分是 matrix-matrix 乘法，对算力利用充分。后来的 chunked prefill 就是利用这点把两者混合调度。

OPT-13B 单 token 的 KV = 2 (K 和 V) × 5120 (hidden) × 40 (层) × 2B (FP16) = 800KB；一条 2048-token 请求最多 1.6GB。

![三种 KV cache 浪费](/images/posts/pagedattention/fig3-waste.png)

在 PagedAttention 出现之前的系统存在 3 种 KV cache 浪费：

- **Reserved**：给未来的 token 留的占用位置。最终会被用到，但是占用期间会挡住别的请求而不让出这一部分位置。
- **内部碎片**：实际生成的 token 远远小于预计生成的 token 数量，浪费槽位。
- **外部碎片**：空闲内存散落在不同位置，总量可能够，但没有足够大的连续区域。

旧系统为什么只能整段预分配？

1. 框架侧：PyTorch 这类深度学习框架当中的算子要求 tensor 在内存中连续存储，所以一条请求的 KV cache 是一块连续的 tensor。
2. 输出的长度先验未知，KV cache 随着 decode 生成动态增长，所以都按一个最大可能的生成长度进行整段的预留。

![内存利用率对比](/images/posts/pagedattention/fig4-utilization.png)

vLLM 将 token 的内存利用率做到 96.3%，剩下的浪费只来自于每条请求最后一个没有填满的 block。

## PagedAttention 设计

![PagedAttention 示意](/images/posts/pagedattention/fig5-paged.png)

算法本身对 attention 的加权求和是完全保留的。

![PagedAttention 公式](/images/posts/pagedattention/fig6-formula.png)

这个公式也就是：PagedAttention = 普通 Attention + 把 KV 按 block 分组读取。**改变内存布局，不改变算法**。

和操作系统当中的 virtual memory 处理方式类似。Block table 记录每个 logical block 对应到哪个 physical block + 填了几个 slot。

![Block table 映射](/images/posts/pagedattention/fig7-blocktable.png)

### decode 一步的全部流程

vLLM 先选出一组 sequences 然后 batch 在一起
→ 为需要新 logical block 的请求分配 physical block
→ 把所有的 input token（prefill 请求的全部 token + decode 请求的最新 1 个 token）拼成一个 batch 给模型
→ PagedAttention kernel 按 block table 读旧 KV，写新 KV

除了消除内存碎片，PagedAttention 还有其他好处。文章中的其他 decoding 场景下的应用介绍了这一部分。

![Parallel sampling 共享](/images/posts/pagedattention/fig8-sampling.png)

- **Parallel Sampling（一个 prompt 里 sample n 次）**：这 n 条序列的 logical block 映射到同一 physical block，同时在 physical block 上增加一个新的属性 ref count。decoding 阶段各自采样各自的 logical block，有序列需要写共享 physical block 时，若 ref count > 1 就触发 copy-on-write（分配新块 → 拷贝 → 原块 ref count 减 1）。

![Beam search 共享](/images/posts/pagedattention/fig9-beam.png)

- **Beam search（共享不止在 prompt）**：类似 OS 里不断 fork 出来的进程树。被淘汰的 candidate 把 ref count 减到 0 然后就可以回收掉这些 block 了。在图中虚线以后，vLLM 之前的系统会复制 candidate 2 的 KV 到 candidate 3 上然后继续生成，有了 vLLM 的 physical block sharing 就可以大大减少这些 copy 的频率。

![Shared prefix](/images/posts/pagedattention/fig10-prefix.png)

- **Shared prefix**：预先把公共的前缀 KV 存到一批 physical block 里。新来的请求直接把 logical block 指过去就可以了。

### 调度与抢占

First-come-first-serve (FCFS)；内存不够时抢占最晚到的请求。

抢占遵循 all-or-nothing eviction policy：要抢占就一次性驱逐一个序列的所有 block。为什么不是 LRU 的原因也很容易看出来——一条序列的所有 block 每一步都会一起被访问，按 block 驱逐没有意义。同一请求下的多个序列（beam candidates）采用 gang-scheduling，因为他们当中存在着 shared block。

恢复有两条路径：

- **Swapping**：被抢占的 blocks 全部 copy 到 CPU RAM 里去。
- **Recomputation**：被抢占时已生成的 token 可以并进 prompt 一次 prefill 算回来。原本逐 token decode 出来的 KV，恢复时变成一次 matrix-matrix 并行算完，所以重算远快于"重新生成一遍"。

### 关于分布式

TP 切 head：每张 GPU 负责一部分 attention heads，KV cache 沿 head 维度被 TP 分片分别放在不同的 GPU 上。所有 TP rank 使用同一套 logical → physical block 映射，也就是说 block table 只有一份。

> 比如说 GPU0 的 slot 7 和 GPU1 的 slot 7 对应同一个 request 的同一段 tokens，但里面存的是不同 head 的 KV。

Scheduler 把 token ids + block table 发下去，worker 之间只做 all-reduce。

## 实现与实验

### kernel 方面的优化

动机：PagedAttention 为了省显存，把 KV cache 切成很多不连续的 block，但"不连续"会让 GPU 访问变碎，变成间接寻址了。所以 vLLM 写了几个专门的 fused kernel，把这些碎操作一次性做掉。

- **Fused reshape and block write**：decode 新生成一个 token 后会发生 `K/V → reshape → 查 block table → 找到应该写入哪个 physical block → 写进 KV cache`。如果很朴素地实现，可能是：

```
Kernel 1：reshape
Kernel 2：计算/整理地址
Kernel 3：写 KV cache
```

每启动一个 CUDA kernel 都有 launch overhead，于是 vLLM 把 reshape + 查地址 + 写 block 融合进一个 fused kernel。

- **Fused block read and attention**：不先把分页 KV 拼起来，而是一边找 block、一边读取、一边算 attention。同时一个 warp 负责读一个 block，这些地址连续，GPU 可以 coalesced memory access。

- **Fused block copy**：Copy-on-write 的行为会经常出现，会有很多小 block 的移动通过 cudaMemcpyAsync API。所以 vLLM 直接把不同 block 的 CoW 行为 fused 进一个 kernel 里实现了。

### 对各种 decoding algorithms 的支持

vLLM 通过实现三个基础的方法：**fork、append、free**。parallel sampling、beam search、prefix sharing 等等都可以拆成这三个基础方法的组合。

---

核心 insight 就一句话：把 OS 虚拟内存的思想搬到 KV cache 管理上。

</div>

<div class="lang lang-en" markdown="1">

![PagedAttention paper](/images/posts/pagedattention/fig1-cover.png)

The 2023 vLLM paper, *Efficient Memory Management for Large Language Model Serving with PagedAttention*, is probably one of the most-cited works in LLM serving — nearly every inference framework today inherits its ideas in one way or another. The question it answers is simple: **how should the KV cache in GPU memory be managed?** The paper's answer borrows the paging idea from operating systems' virtual memory: instead of requiring contiguous storage, the KV cache is split into fixed-size blocks, with a block table mapping logical blocks to physical ones. This single change raised memory utilization from under 40% to above 96% and improved throughput by 2-4×.

This post is my personal understanding of the paper, following its structure in three steps: **why KV cache management directly determines throughput (motivation) → how PagedAttention is designed (paging, sharing, scheduling) → how vLLM implements it (kernel optimizations)**.

## Motivation: KV cache management = throughput

When serving, handling multiple requests at once (i.e., increasing the batch size) obviously increases system throughput.

![GPU memory breakdown](/images/posts/pagedattention/fig2-memory.png)

Running a 13B model on an A100 40G, the KV cache takes about 30% of memory. So the question of how many requests can be served concurrently reduces to how the KV cache is managed.

Since the decode phase of an autoregressive LLM generates one token at a time, each step is merely a matrix-vector multiplication, which underutilizes GPU compute. The way to raise throughput is to increase the batch size, amortizing each weight-loading pass over many requests. But batch size can't grow arbitrarily — its ceiling is set by KV cache management. **Memory efficiency is throughput**; this phase is memory bound.

Meanwhile, the prefill phase is matrix-matrix multiplication and uses compute well. The later chunked prefill technique exploits exactly this by mixing the two in one schedule.

For OPT-13B, the KV per token = 2 (K and V) × 5120 (hidden) × 40 (layers) × 2B (FP16) = 800KB; a 2048-token request can take up to 1.6GB.

![Three kinds of KV cache waste](/images/posts/pagedattention/fig3-waste.png)

Systems before PagedAttention suffered three kinds of KV cache waste:

- **Reserved**: slots held for future tokens. They are eventually used, but while held they block other requests from using that memory.
- **Internal fragmentation**: the actual number of generated tokens is far smaller than the reserved maximum, wasting slots.
- **External fragmentation**: free memory scattered across locations — the total may suffice, but no contiguous region is large enough.

Why could old systems only pre-allocate one contiguous chunk?

1. Framework side: operators in deep learning frameworks like PyTorch require tensors to be contiguous in memory, so one request's KV cache is a single contiguous tensor.
2. The output length is unknown a priori, and the KV cache grows dynamically during decoding — so systems reserve one chunk sized for the maximum possible length.

![Memory utilization comparison](/images/posts/pagedattention/fig4-utilization.png)

vLLM pushes token memory utilization to 96.3%; the only remaining waste comes from each request's last, partially-filled block.

## PagedAttention design

![PagedAttention illustration](/images/posts/pagedattention/fig5-paged.png)

The algorithm fully preserves attention's weighted sum.

![PagedAttention formula](/images/posts/pagedattention/fig6-formula.png)

The formula amounts to: PagedAttention = ordinary attention + reading KV in block-sized groups. **The memory layout changes; the algorithm doesn't.**

This mirrors virtual memory in operating systems. The block table records which physical block each logical block maps to, plus how many slots are filled.

![Block table mapping](/images/posts/pagedattention/fig7-blocktable.png)

### One decode step, end to end

vLLM picks a group of sequences and batches them together
→ allocates physical blocks for requests that need a new logical block
→ concatenates all input tokens (all tokens of prefill requests + the latest single token of decode requests) into one batch for the model
→ the PagedAttention kernel reads old KV and writes new KV via the block table.

Beyond eliminating fragmentation, PagedAttention brings other benefits, covered in the paper's section on other decoding scenarios.

![Parallel sampling sharing](/images/posts/pagedattention/fig8-sampling.png)

- **Parallel sampling (sampling n times from one prompt)**: the n sequences' logical blocks map to the same physical block, and each physical block gains a new attribute — a ref count. During decoding each sequence samples through its own logical blocks; when a sequence needs to write to a shared physical block and the ref count > 1, copy-on-write triggers (allocate a new block → copy → decrement the original block's ref count).

![Beam search sharing](/images/posts/pagedattention/fig9-beam.png)

- **Beam search (sharing beyond the prompt)**: like a process tree forked repeatedly in an OS. When a candidate is pruned, its ref counts drop to 0 and the blocks can be reclaimed. After the dashed line in the figure, pre-vLLM systems would copy candidate 2's KV to candidate 3 and continue generating; with vLLM's physical block sharing, such copies become far less frequent.

![Shared prefix](/images/posts/pagedattention/fig10-prefix.png)

- **Shared prefix**: store the common prefix's KV in a set of physical blocks ahead of time. New requests simply point their logical blocks at them.

### Scheduling and preemption

First-come-first-serve (FCFS); when memory runs out, the latest-arriving request is preempted.

Preemption follows an all-or-nothing eviction policy: evicting a sequence means evicting all of its blocks at once. It's easy to see why LRU wouldn't help — all blocks of a sequence are accessed together at every step, so per-block eviction is meaningless. Multiple sequences of the same request (beam candidates) are gang-scheduled, because they hold shared blocks.

There are two recovery paths:

- **Swapping**: copy all preempted blocks out to CPU RAM.
- **Recomputation**: the tokens already generated at preemption time can be merged into the prompt and recomputed in a single prefill. KV that was originally produced token-by-token during decode is recomputed in one parallel matrix-matrix pass — so recomputation is far faster than "generating again."

### On distributed serving

TP shards by head: each GPU handles a subset of attention heads, and the KV cache is sharded along the head dimension across GPUs. All TP ranks use the same logical → physical block mapping — there is only one block table.

> For example, slot 7 on GPU0 and slot 7 on GPU1 correspond to the same span of tokens of the same request, but store the KV of different heads.

The scheduler broadcasts token ids + the block table; workers only need all-reduce among themselves.

## Implementation and experiments

### Kernel optimizations

Motivation: to save memory, PagedAttention splits the KV cache into many non-contiguous blocks, but "non-contiguous" fragments GPU access into indirect addressing. So vLLM wrote several dedicated fused kernels to fold these scattered operations together.

- **Fused reshape and block write**: after decoding one new token, the sequence is `K/V → reshape → look up the block table → find which physical block to write → write into the KV cache`. A naive implementation might be:

```
Kernel 1: reshape
Kernel 2: compute/arrange addresses
Kernel 3: write KV cache
```

Every CUDA kernel launch has overhead, so vLLM fuses reshape + address lookup + block write into one kernel.

- **Fused block read and attention**: rather than first assembling the paged KV, it locates blocks, reads, and computes attention all at once. One warp reads one block; those addresses are contiguous, enabling coalesced memory access.

- **Fused block copy**: copy-on-write happens frequently, producing many small block moves through the cudaMemcpyAsync API. vLLM fuses these per-block CoW operations into a single kernel.

### Supporting various decoding algorithms

vLLM implements three primitive methods: **fork, append, free**. Parallel sampling, beam search, prefix sharing, and more can all be decomposed into combinations of these three.

---

The core insight in one sentence: bring the OS virtual-memory idea into KV cache management.

</div>
