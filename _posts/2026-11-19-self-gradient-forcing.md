---
layout: post
categories: [diffusion-models]
title: "12. Self Gradient Forcing: Learning to Write Better Video Memory"
date: 2026-11-19 12:00:00 +0100
description: "How context gradients help an autoregressive video model learn to write more useful KV memory."
math: true
---

[Self Gradient Forcing: Native Long Video Extrapolation](https://arxiv.org/abs/2607.20368)

Self Forcing trains an autoregressive video generator on its own generated history. This reduces the gap between training and inference: at inference, the model also has to continue from frames it generated itself.

However, the authors of Self Gradient Forcing (SGF) point out another gap. During Self Forcing training, the historical KV cache is detached. A loss on a later chunk can train the model to **use** the cached history, but it cannot directly supervise the earlier computation that **wrote** that history into the cache.

SGF addresses this with two training passes. The first generates a video without tracking gradients. The second reconstructs the relevant computation with gradients enabled, allowing future losses to improve how generated history is encoded as keys and values. [Self Gradient Forcing, Introduction](https://arxiv.org/html/2607.20368v2#S1)

## Background: Self Forcing and the KV Cache

An autoregressive video model generates a video one latent frame or chunk at a time. Each new chunk depends on earlier generated chunks. To avoid recomputing all the previous attention states, the model keeps their **keys and values** in a KV cache.

For example, consider three chunks:

1. Generate chunk 1 and write its keys and values into the cache.
2. Generate chunk 2 while attending to the cache, then add its keys and values.
3. Generate chunk 3 while attending to the cached history.

The cached entries are representations of the generated latent history, not the video frames themselves. In the paper's setting, a generated clean latent is processed at the clean context timestep to produce its cached representations.

Self Forcing uses the student's own generated chunks as history during training. This matters because the student must handle its own imperfect outputs at inference, too. Its distribution-matching loss then provides a training signal for the generated video. [Self Gradient Forcing, Section 3.1](https://arxiv.org/html/2607.20368v2#S3.SS1)

## The Historical Context-Gradient Gap

Although Self Forcing uses generated history, the historical KV entries are treated as detached state. Consider the loss on chunk 3:

```text
chunk 1 latent → write K/V → cached history → generate chunk 3 → loss
                                  ↑
                           gradient stops here
```

The loss can still update the computations that generate chunk 3 and read the cache. But it cannot travel through the detached cache to directly tell the model how the earlier clean-context computation should have written chunk 1's keys and values.

This distinction is important because the same model parameters are used for both denoising and context encoding. A training update can indirectly change how history is written, but detached-cache training does not provide the missing **future-loss gradient through that writing path**. The authors call this the *historical context-gradient gap*.

Imagine that chunk 1 shows a person wearing a red coat, and chunk 3 depends on that history. If the cached representation does not preserve the coat information well, a loss on chunk 3 can teach the model how to use the available cache. With a detached cache, that loss cannot directly teach the earlier context encoder to write a more useful representation of the coat.

The problem is therefore not that Self Forcing ignores its history. It is that future losses cannot directly supervise one particular path: **how self-generated history becomes memory**. [Self Gradient Forcing, Section 3.1](https://arxiv.org/html/2607.20368v2#S3.SS1)

## Why Not Simply Keep the Whole Cache Differentiable?

One possible solution is to keep the computation graph for every cache write during the autoregressive rollout. Later losses could then backpropagate through the earlier cached representations.

The difficulty is memory. The model repeatedly denoises chunks and writes and reads cache entries across many layers. Keeping those earlier computations available for backpropagation requires storing their intermediate activations. The training graph can become expensive as the rollout grows.

SGF instead reconstructs the relevant computation in a second pass. This opens a gradient path through context writing without keeping the complete serial rollout graph. The extra pass still has a training cost; it is not free. [Self Gradient Forcing, Sections 3.2–3.3](https://arxiv.org/html/2607.20368v2#S3.SS2)

## Self Gradient Forcing: Two Passes

### Pass 1: Generate and Record the History

The first pass performs a serial autoregressive rollout with gradients disabled:

1. Sample a denoising exit step for the training rollout.
2. Generate the latent chunks one after another, using the KV cache as history.
3. For each chunk, record both the noisy latent input at the selected step and the corresponding predicted clean latent.
4. Use the predicted clean latent to update the cache for later chunks.

The selected step gives the second pass a particular denoising computation to reconstruct. The recorded clean latents provide the self-generated history, while the recorded noisy latents provide the inputs whose predictions will be trained.

Because this rollout runs without gradients, the model does not need to retain its full serial computation graph. [Self Gradient Forcing, Section 3.3](https://arxiv.org/html/2607.20368v2#S3.SS3)

### Pass 2: Reconstruct with Context Gradients

The second pass takes the recorded latents and reconstructs the selected denoising computation in parallel:

1. Treat the recorded clean context latents as fixed inputs.
2. Run the model on those context latents again at the clean context timestep to recompute their keys and values.
3. Reconstruct predictions for the recorded noisy inputs, using a causal attention mask so each chunk sees only the history available to it in the rollout.
4. Apply the DMD loss to the reconstructed predictions and backpropagate through the reconstructed context-writing and attention computations.

The important detail is that the **context latents are detached, but the recomputed keys and values are not**. This lets the loss train how the model encodes a given history into memory, without backpropagating through the earlier denoising trajectory that originally produced that history.

In a simplified picture:

```text
fixed generated history → recompute K/V → future prediction → DMD loss
                                 ← gradient flows ←
```

The reconstruction mask must match the relevant causal history and cache policy of the serial rollout. Otherwise, the second pass would train on a different attention context. [Self Gradient Forcing, Section 3.3](https://arxiv.org/html/2607.20368v2#S3.SS3)

## Where Does the Gradient Go?

Let $$\tilde{x}_{\mathrm{ctx}}$$ be a generated context latent and $$C_\theta$$ the computation that encodes context into keys and values. SGF reconstructs the memory as

$$
\mathrm{KV}_{\mathrm{ctx}}
=C_\theta(\operatorname{stopgrad}(\tilde{x}_{\mathrm{ctx}})).
$$

The input latent is fixed, but $$C_\theta$$ still depends on the model parameters $$\theta$$. When a future prediction uses this memory, its DMD loss has a gradient path through the reconstructed keys and values:

$$
\nabla_\theta\mathcal L_{\mathrm{DMD}}
\;\supset\;
\frac{\partial\mathcal L_{\mathrm{DMD}}}{\partial\mathrm{KV}_{\mathrm{ctx}}}
\frac{\partial\mathrm{KV}_{\mathrm{ctx}}}{\partial\theta}.
$$

The symbol $$\supset$$ emphasizes that this is **one part** of the gradient. The loss also trains the computation that reads the history and predicts the future chunk.

SGF does not backpropagate through the full rollout, the generation of the recorded context latents, or the original serial cache updates. It restores the context-writing gradient **within the reconstructed training window**, rather than calculating a full gradient through the entire generated video. [Self Gradient Forcing, Section 3.4](https://arxiv.org/html/2607.20368v2#S3.SS4)

## Self Forcing vs. Self Gradient Forcing

| | Self Forcing | Self Gradient Forcing |
|---|---|---|
| Training history | Student-generated history | Student-generated history |
| Future loss trains history reading | Yes | Yes |
| Future loss flows through context KV writing | No, the historical cache is detached | Yes, through reconstructed context K/V |
| Full rollout backpropagation | No | No |

The central change is therefore **not a new source of history**: both methods use the student's own rollout. SGF changes which computations receive gradients from the future loss.


## Training Window in the Released Code

In the authors' released configurations, each training video contains **21 latent frames**. The number of chunks depends on how many latent frames the student generates together:

- **Frame-wise:** 1 latent frame per chunk, so Pass 1 generates and Pass 2 reconstructs **21 chunks**.
- **Chunk-wise:** 3 latent frames per chunk, so Pass 1 generates and Pass 2 reconstructs **7 chunks**.

For the chunk-wise setting, we can picture the two passes as:

```text
Pass 1: generate 7 chunks sequentially (3 latent frames each)
                              ↓
Pass 2: reconstruct predictions for all 7 chunks in parallel
                              ↓
        apply the DMD training signal to the reconstructed video
```

These are frames in the video model's **latent space**, not decoded pixel frames. Pass 2 reconstructs the predictions at **one sampled denoising step** across the training window; it does not reconstruct every step of each chunk's denoising trajectory. The numbers are configuration choices, not fixed requirements of SGF. [Frame-wise configuration](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/main/configs/self_gradient_forcing_framewise.yaml) · [Chunk-wise configuration](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/main/configs/self_gradient_forcing_chunkwise.yaml) · [Training pipeline](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/main/pipeline/self_gradient_forcing_training.py)

## Python-Style Pseudocode

The following pseudocode shows one **student update**. It follows the paper's two-pass algorithm, with implementation details simplified. `student`, `teacher`, and `fake_model` are already initialized; the teacher is frozen, and the fake model is trained separately to estimate the student's score.

```python
# The same exit step is used for all chunks in this example.
exit_step = random_choice(denoising_steps)
cache = empty_kv_cache()
recorded_noisy = []
recorded_context = []

# ---------------------------------------------------------
# Pass 1: generate the student's own history, without gradients
# ---------------------------------------------------------
with torch.no_grad():
    for chunk in range(number_of_chunks):
        noisy = torch.randn(latent_shape)

        for t in denoising_steps:
            # Predict a clean latent using the history in the cache.
            clean = student.predict_clean(noisy, t, cache, prompt)

            if t == exit_step:
                # Save the input and output of this denoising step.
                recorded_noisy.append(noisy)
                recorded_context.append(clean)
                break

            # Add noise to the clean prediction for the next step.
            next_t = next_denoising_step(t)
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, next_t)

        # Encode the generated clean latent at the clean context timestep.
        # Later chunks use this cache as their generated history.
        cache = student.write_context(clean, cache, prompt)

# ---------------------------------------------------------
# Pass 2: reconstruct the selected step, with gradients
# ---------------------------------------------------------
# The recorded latent values are fixed; no gradient goes back to Pass 1.
context = torch.stack(recorded_context).detach()
noisy_inputs = torch.stack(recorded_noisy).detach()

# In one causal reconstruction, the student:
# 1. re-encodes the fixed context at the clean timestep,
# 2. computes new K/V representations with gradients enabled,
# 3. predicts each chunk from its noisy input and allowed past context.
predicted_video = student.reconstruct_parallel(
    noisy_inputs, context, exit_step, causal_cache_mask, prompt
)

# ---------------------------------------------------------
# Use a DMD gradient to update the student
# ---------------------------------------------------------
with torch.no_grad():
    t = sample_dmd_timestep()
    noise = torch.randn_like(predicted_video)
    noisy_video = scheduler.add_noise(predicted_video, noise, t)

    # Both scores are evaluated on the same noisy generated video.
    real_score = teacher.score(noisy_video, t, prompt)
    fake_score = fake_model.score(noisy_video, t, prompt)
    dmd_direction = fake_score - real_score

# A detached MSE target applies the score-difference direction to the output.
target = (predicted_video - dmd_direction).detach()
loss = 0.5 * ((predicted_video - target) ** 2).mean()

student_optimizer.zero_grad()
loss.backward()
student_optimizer.step()
```

The `student.reconstruct_parallel` call represents the paper's parallel, causally masked reconstruction: a chunk's prediction may read its allowed earlier context, not its own or future clean context. The mask also needs to match the rollout's window and sink policy. The method names above describe operations; they are not necessarily the names used in the authors' code.

The DMD part is schematic: actual training includes timestep-dependent factors and practical normalization. The detached MSE target is a **surrogate for applying the approximate DMD direction**, not the KL divergence itself. The fake-score model also needs a separate denoising update on student-generated samples; that update is omitted here to keep the focus on SGF.

The key line is the reconstruction: although `context` is detached, the student recomputes its keys and values **with gradients enabled**. The DMD loss can therefore update the context-writing computation, but cannot backpropagate through the Pass 1 rollout. [Self Gradient Forcing, Algorithm 1](https://arxiv.org/html/2607.20368v2#S3.SS3)
