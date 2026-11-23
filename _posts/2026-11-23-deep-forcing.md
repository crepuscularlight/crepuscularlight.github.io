---
layout: post
categories: [diffusion-models]
title: "14. Deep Forcing: Better Memory for Long Video Generation"
date: 2026-11-23 03:00:00 +0100
description: "How Deep Sink and participative cache compression support long video generation without retraining."
math: true
---

[Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression](https://arxiv.org/abs/2512.05081)

Self Forcing trains an autoregressive video model on its own generated history. This reduces the gap between training and inference, but long video generation can still suffer from error accumulation: colors may drift, motion may slow down, and the scene may gradually lose quality.

Deep Forcing studies whether better management of the model's **KV cache** can help. Inspired by attention sinks in StreamingLLM, it keeps a larger portion of the initial context as an anchor. It also selectively retains useful information from later frames instead of discarding history based only on age.

The method has two components:

1. **Deep Sink:** preserve a relatively large initial context and adjust its temporal position encoding as generation continues.
2. **Participative Compression:** select useful intermediate tokens based on attention from recent queries, while keeping the cache size bounded.

An important distinction is that Deep Forcing is **training-free**: it changes how an existing model manages its cache during generation, without fine-tuning the model or learning a new compression network. [Deep Forcing, Introduction](https://arxiv.org/html/2512.05081#S1)

## Background: The KV Cache and Attention Sinks

An autoregressive video model generates latent frames or chunks one after another. To generate a new chunk, it attends to information from previous chunks. A KV cache stores the previous chunks' **keys and values**, so the model can reuse them during attention.

A long video may contain far more history than the model can efficiently keep. A sliding window limits the cache size by removing old entries as new ones arrive. This keeps memory bounded, but eventually the model loses direct access to the beginning of the video.

An **attention sink** keeps some early keys and values in the cache even after they would normally be removed. The idea is related to StreamingLLM, where retaining initial tokens helps stabilize attention over long sequences. For video, these early representations can also provide a reference for the subject and scene.

However, the paper finds that directly adopting a small, LLM-style sink does not work as well for its video setting. Video models can attend to information distributed across multiple early frames, not just a few initial tokens. [Deep Forcing, Sections 2 and 4.2](https://arxiv.org/html/2512.05081#S4.SS2)

## Preliminary: RoPE and 3D RoPE

Attention needs information about position. **Rotary Position Embedding (RoPE)** adds this information by rotating query and key vectors according to their positions before their attention scores are computed.

For one pair of vector coordinates, the rotation at position $$p$$ can be written as

$$
R(p)=
\begin{pmatrix}
\cos(p\omega) & -\sin(p\omega)\\
\sin(p\omega) & \cos(p\omega)
\end{pmatrix},
$$

where $$\omega$$ controls the rotation frequency. Real models use many coordinate pairs and frequencies, but the basic idea is the same.

If a query is at position $$m$$ and a key is at position $$n$$, their rotated dot product is

$$
(R(m)q)^\top(R(n)k)=q^\top R(n-m)k.
$$

This lets the query–key comparison depend on their **relative position** $$n-m$$ as well as their content.

### From 1D Positions to a Video Grid

A video is represented as latent frames and divided into spatial patches. We can think of these patches as a three-dimensional grid of tokens:

```text
        ┌──────────────┐
        │ latent t = 0 │  → width
        └──────────────┘
               ↓ time
        ┌──────────────┐
        │ latent t = 1 │
        └──────────────┘
```

Each token has a coordinate $$(t,h,w)$$: its position in **time**, **height**, and **width**. These are coordinates in the latent token grid, not necessarily individual video pixels. For example:

```text
A: (t=0, h=2, w=3)  → an early latent frame
B: (t=5, h=2, w=3)  → a later latent frame, same spatial location
C: (t=5, h=4, w=3)  → a later latent frame, different height
```

### How Does 3D RoPE Work?

In one common design, the query and key vectors are divided into parts for the three axes:

$$
q=[q_t,q_h,q_w],\qquad k=[k_t,k_h,k_w].
$$

For a token at $$(t,h,w)$$, each part is rotated according to the corresponding coordinate:

$$
\operatorname{RoPE}_{3D}(q;t,h,w)
=\left[R_t(t)q_t,\;R_h(h)q_h,\;R_w(w)q_w\right].
$$

The key is rotated in the same way. The model can then use relative temporal and spatial positions when comparing a query with a key. For example, it can distinguish a token at the same location in an earlier frame from a token above it in the current frame. The attention mask still determines which positions are allowed to interact.

The three parts do not have to be equal in size. This is a model design choice; the [Wan2.1 implementation](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py) allocates frequency dimensions across the temporal and spatial axes.

### Why Does This Matter for Deep Forcing?

Deep Forcing keeps early tokens in the KV cache as a sink. Later in a long video, a current query can have a temporal position far away from the cached sink keys:

```text
Sink key:      temporal position 1
Current query: temporal position 200
```

The temporal RoPE component represents that large gap. The paper argues that this can be problematic when the model was trained on much shorter contexts. Deep Forcing adjusts the **temporal RoPE phase of cached keys** to fit the current cache timeline, while preserving their height and width position information:

```text
Before: (time=1,        height=2, width=3)
After:  (time=adjusted, height=2, width=3)
```

This changes the temporal position presented to attention. It does not move the token within the image or regenerate the original frame. The next section explains how this adjustment is used with Deep Sink. [Deep Forcing, Section 4.2](https://arxiv.org/html/2512.05081#S4.SS2)

## Deep Sink: Keep More Initial Context

Deep Sink reserves a relatively large part of the context budget for initial frames. Rather than keeping only a few initial tokens, it preserves a longer early segment as a persistent reference.

For intuition, imagine a cache split like this:

```text
[ early frames kept as Deep Sink ][ later context ][ recent frames ]
```

The authors observe that attention in their video model is distributed over both early and intermediate frames. Keeping a larger initial segment can therefore retain more useful scene and subject information during a long rollout. In their experiments, a sink occupying roughly half of the available context works well.

Bigger is not always better. If too much of the cache is reserved for the beginning, less space remains for recent changes and the video can become repetitive. The paper's sink-size ablation shows this trade-off. [Deep Forcing, Section 4.2](https://arxiv.org/html/2512.05081#S4.SS2)

### Why Adjust Temporal RoPE?

As generation continues, the initial sink remains in the cache while the current frame moves far forward in time. If the original temporal positions are used directly, attention may have to compare a very early key with a much later query, creating a positional gap unlike the short contexts the model was trained on.

Deep Sink adjusts the **temporal RoPE phase** of cached sink keys to align them with the current cache timeline. Conceptually:

```text
Original positions:  [ early sink ] ............... [ recent frames ]
Adjusted positions:  [ early sink ][ ... ][ recent frames ]
```

The adjustment changes the temporal position used by attention; it does not regenerate the initial frames or change when they actually occurred. The spatial components are kept intact. The goal is to let the model use the initial anchor without presenting an excessively large positional gap. [Deep Forcing, Section 4.2](https://arxiv.org/html/2512.05081#S4.SS2)

## Why Is a Large Sink Not Enough?

A Deep Sink preserves the beginning and a local window preserves recent context. But the middle of a long rollout can contain useful details too: an object may enter the scene, a character may change pose, or the camera may move.

A simple first-in, first-out policy removes the oldest rolling entries regardless of their content. It can discard useful intermediate information while keeping other entries that are less relevant to the current generation.

Participative Compression addresses this by asking a different question: **which cached tokens are useful to the model's recent attention?** It does not learn to compress the history into a new latent. Instead, it selects and retains existing KV tokens according to an importance score. [Deep Forcing, Section 4.3](https://arxiv.org/html/2512.05081#S4.SS3)

## Participative Compression

When the cache needs to be compressed, the method conceptually separates its contents into three groups:

1. **Sink:** early tokens that remain protected.
2. **Recent:** the latest context, kept for local continuity.
3. **Candidates:** intermediate tokens that can be selected or removed.

```text
Before compression:
[ sink ][ candidate intermediate tokens ][ recent ]

After compression:
[ sink ][ selected important tokens     ][ recent ]
```

The method compares recent queries with the candidate keys. A candidate that receives a higher aggregated relevance score is more likely to be retained. The paper writes a simplified importance score as

$$
\phi_j=\sum_r q_r^\top k_j,
$$

where $$q_r$$ is a recent query and $$k_j$$ is a candidate key. The score $$\phi_j$$ summarizes how strongly those queries relate to candidate $$j$$. The method keeps the highest-scoring candidates, together with their corresponding values, within a fixed cache budget.

For example, if recent queries strongly relate to tokens describing a moving person, some of those tokens may be kept even after other intermediate tokens are removed. This is an intuition, not a guarantee that attention scores always identify the most semantically important or least degraded content.

The authors also adjust the temporal RoPE positions of selected tokens so that the compressed cache has a coherent positional layout. The selected tokens still retain their spatial position information. [Deep Forcing, Section 4.3](https://arxiv.org/html/2512.05081#S4.SS3)

### How Does This Relate to the Cache Size?

The paper's reported configuration uses a sink size of 10, a compressed cache budget of 16, and 4 recent frames. In frame-equivalent terms, this leaves room for approximately 2 frames' worth of selected intermediate tokens:

```text
16-frame-equivalent budget = 10 sink + 2 selected + 4 recent
```

The selection itself operates at the **token level**, so those selected tokens need not form two complete consecutive latent frames. These values describe the authors' experimental setting, not a requirement for every model or cache size. [Deep Forcing, Section 5.1](https://arxiv.org/html/2512.05081#S5.SS1)

## Putting the Two Components Together

During generation, Deep Forcing manages history in three complementary ways:

- **Deep Sink** keeps a persistent reference to the beginning of the video.
- **Participative Compression** keeps selected intermediate tokens that are relevant to recent attention.
- **Recent context** keeps the most recent information available for local motion and continuity.

Both the sink and selected tokens receive temporal RoPE adjustments to fit the cache's positional layout. No additional student, teacher, loss, or model training is required for these cache-management operations.

Compression has a computational cost. The authors apply it only when the window is full and at the first denoising step, and report nearly unchanged average throughput in their measured setting. The exact cost depends on the model, implementation, hardware, and cache configuration. [Deep Forcing, Section 4.3 and Appendix E](https://arxiv.org/html/2512.05081#A5)


## Python-Style Pseudocode

The pseudocode below focuses on **one cache compression event in one attention layer**. It follows the paper's description: protect the sink and recent context, score the intermediate candidates, and keep the most relevant ones. The actual implementation handles multiple layers, attention heads, token layouts, and positional indices.

```python
# At the first denoising step for a new chunk, while computing attention:
#
# cache.keys, cache.values: stored keys and values for this layer
# recent_queries: queries from recent context and the current denoising chunk
# S: number of protected sink tokens
# R: number of protected recent tokens
# C: number of intermediate candidate tokens to keep

if cache_is_full and timestep == first_denoising_step:
    # 1. Split the cache. S and R are token counts in this example.
    sink_keys = cache.keys[:S]
    sink_values = cache.values[:S]

    candidate_keys = cache.keys[S:-R]
    candidate_values = cache.values[S:-R]

    recent_keys = cache.keys[-R:]
    recent_values = cache.values[-R:]

    # 2. Score each candidate using its dot products with recent queries.
    #    Each column corresponds to one candidate key.
    similarity = recent_queries @ candidate_keys.T
    importance = similarity.sum(dim=0)

    # 3. Retain the C highest-scoring candidates and their matching values.
    selected_indices = importance.topk(C).indices
    selected_indices = selected_indices.sort().values  # preserve cache order
    selected_keys = candidate_keys[selected_indices]
    selected_values = candidate_values[selected_indices]

    # 4. Shift only the temporal RoPE phase of cached keys.
    #    This represents a position adjustment, not a change to image content.
    sink_keys = adjust_temporal_rope(sink_keys, sink_time_shift)
    selected_keys = adjust_temporal_rope(selected_keys, selected_time_shifts)

    # 5. Rebuild the cache for attention and later denoising steps.
    cache.keys = concatenate(sink_keys, selected_keys, recent_keys)
    cache.values = concatenate(sink_values, selected_values, recent_values)
```

`adjust_temporal_rope` rotates the temporal part of each already-positioned key by the required position shift; its spatial part is left unchanged. `concatenate` joins token sequences in cache order. The shifts depend on the token positions and the desired compressed-cache layout, so they are represented symbolically here. Sink keys also need the appropriate temporal alignment as generation advances; the snippet shows the adjustment during a compression event.

In the paper's reported configuration, the **frame-equivalent** budget is 10 sink + 2 selected + 4 recent = 16. The pseudocode uses token counts instead: selection can keep individual tokens rather than two complete frames. If there are too few candidates, an implementation must keep only those available.

Compression is performed independently for each layer when the window is full, at the first denoising step. The selected cache is then used for the following denoising steps. The authors combine queries from clean recent context and the current denoising input; the code above groups them as `recent_queries`. This is an explanatory sketch, not runnable repository code. [Deep Forcing, Algorithm 1 and Appendix C](https://arxiv.org/html/2512.05081#S4.SS3)
