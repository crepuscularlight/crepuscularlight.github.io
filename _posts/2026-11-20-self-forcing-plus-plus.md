---
layout: post
categories: [diffusion-models]
title: "13. Self-Forcing++: Learning from Long Video Rollouts"
date: 2026-11-20 12:00:00 +0100
description: "How a short-video teacher guides a student on windows sampled from long autoregressive rollouts."
math: true
---

[Self-Forcing++: Towards Minute-Scale High-Quality Video Generation](https://arxiv.org/abs/2510.02283)

Self Forcing trains an autoregressive video generator using its own generated history. This reduces the difference between training and inference: in both cases, the model must continue from frames it generated itself.

However, Self Forcing can still struggle when generating much longer videos than it saw during training. In the setting studied by Self-Forcing++, the teacher is trained on approximately five-second clips, and the student's distillation rollout is also limited to a short horizon. During a much longer rollout, small mistakes can accumulate: the video may lose motion, become increasingly degraded, or freeze.

Self-Forcing++ asks: **can a short-video teacher help train a student on the problems that appear much later in a long video?**

Its central idea is to let the student generate a long video, sample a short window from that rollout, and use the teacher to guide distribution matching on the sampled window. The student therefore encounters its own long-rollout errors during training, even though the teacher only needs to process a short segment. [Self-Forcing++, Sections 1 and 3.2](https://openreview.net/pdf/be9702647edc6df7220cf7f97f1d6672f0e68fa4.pdf)

## Why Can Self Forcing Still Accumulate Errors?

In Self Forcing, each new chunk is generated using the student's own previous outputs. This is important because, at inference, the student cannot rely on perfect ground-truth history.

But using self-generated history does not necessarily expose the student to every situation it will encounter later. Imagine training only on the first five seconds of a generated video:

```text
Training:  [ first 5 seconds ]
Inference: [ first 5 seconds ][ ... ][ much later in the video ]
```

At the beginning of a rollout, the history has only a short time to accumulate mistakes. After many more chunks, small errors can become part of the context for future generation. The student may then enter states that were rarely encountered during short-rollout training.

The short teacher horizon is part of the challenge, but it is not enough by itself to explain every long-video failure. The more specific issue is that **the student is trained and supervised over a shorter rollout than it must handle at inference**. The paper focuses on exposing the student to those later, potentially degraded states. [Self-Forcing++, Section 3.2](https://openreview.net/pdf/be9702647edc6df7220cf7f97f1d6672f0e68fa4.pdf)

## The Main Idea: Long Rollout, Short Teacher Window

A short-video teacher does not need to generate or evaluate the entire long video at once. Self-Forcing++ instead uses it to provide guidance on a short contiguous segment sampled from the student's long rollout.

The training idea is:

1. Let the student generate a long video autoregressively using its own history.
2. Sample a short window from somewhere in the generated video, including positions beyond the teacher's usual generation horizon.
3. Add noise to the sampled student latents and use distribution matching to guide the student with the teacher.
4. Repeat training with windows sampled from different positions.

For example, a teacher trained on five-second clips can guide a five-second window sampled from a student's much longer video. The window may contain accumulated errors that do not usually appear at the beginning of a rollout.

The teacher is not suddenly generating a complete minute-long video. It supplies a **local distribution-matching signal** for a short segment. Repeating this at different positions helps the student learn from long-rollout states without requiring a long-video teacher or a long-video training dataset. [Self-Forcing++, Section 3.2](https://openreview.net/pdf/be9702647edc6df7220cf7f97f1d6672f0e68fa4.pdf)

## Backward Noise Initialization

How can the teacher provide useful guidance for a segment that appears in the middle of the student's video?

Starting a new segment from unrelated random noise would not preserve information about the student's existing trajectory. Instead, the method takes generated latent frames and adds noise back to them. For a generated clean latent $$x_0$$, the paper writes this operation in its flow-matching notation as

$$
x_t=(1-\sigma_t)x_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Here, $$x_t$$ is the noisy latent at noise level $$t$$, and $$\sigma_t$$ controls how much noise is added. The resulting noisy input is related to the student's generated segment rather than being an unrelated sample. At very high noise levels, however, little or no information about the original segment may remain.

The student and teacher can then be evaluated on noisy states associated with the selected rollout window. The teacher's knowledge of short-video distributions provides the guidance used for distillation. This does **not** mean the teacher directly repairs and replaces every frame in the long video; the correction is transferred to the student through training. [Self-Forcing++, Section 3.2]

## Extended Distribution Matching

Recall that DMD updates a student using the difference between two score estimates:

- The **real score** comes from the frozen teacher and describes the teacher's target distribution.
- The **fake score** is estimated by a separately trained model and describes the student's current distribution.

Self-Forcing++ extends where this signal is applied. Let the student generate $$N$$ latent frames, and let $$W_i$$ be a contiguous window of $$K$$ frames starting at position $$i$$. A useful schematic view is

$$
i\sim\operatorname{Uniform}\{1,\ldots,N-K+1\},
\qquad
W_i=(x_i,\ldots,x_{i+K-1}).
$$

The teacher and fake model provide score estimates for noisy versions of the sampled window. The student's update is guided by their difference:

$$
\widehat{\nabla_\theta\mathcal L_{\mathrm{extended\text{-}DMD}}}
\;\propto\;
\mathbb E_{i,t}\left[
J_G^\top\left(s_{\mathrm{fake}}(W_{i,t},t)
-s_{\mathrm{real}}(W_{i,t},t)\right)
\right].
$$

Here $$W_{i,t}$$ is the sampled window after adding noise, and $$J_G$$ represents the gradient path from the trainable student predictions back to its parameters. This equation is a simplified illustration: actual training uses estimated scores and timestep-dependent details, so it should not be read as an exactly evaluated KL loss.

The key change from short-horizon DMD is the **sampling position**. The student can receive a distribution-matching signal on a window that occurs much later in its own rollout. This exposes training to errors that develop over time, while keeping the teacher's input within a manageable short window. [Self-Forcing++, Section 3.2 and Appendix A.5]

## Rolling KV Cache During Training and Inference

During autoregressive generation, a KV cache stores attention representations of earlier latent frames. A fixed-size rolling cache removes older entries as new ones arrive, letting the model continue generating without allowing the cache to grow indefinitely.

The paper also identifies a mismatch in its Self Forcing baseline: the baseline uses a fixed cache during short training rollouts but a rolling cache for long inference. Self-Forcing++ uses a rolling KV cache during **both** long training rollouts and inference. This lets the student encounter the cache behavior it will use when generating longer videos.

The rolling cache helps make long generation practical; extended DMD provides the training signal on sampled windows. They address related but different issues. The cache also has a limited context, so it cannot by itself guarantee that every detail from the beginning of a long video will be remembered. [Self-Forcing++, Section 3.2]
## Optional GRPO for Smoother Motion

A rolling attention window can gradually lose older context. Even when individual frames look good, a long video may still contain abrupt scene changes or objects appearing and disappearing unexpectedly.

The paper also explores **Group Relative Policy Optimization (GRPO)** as a further training step. It uses a reward based on optical flow as a proxy for motion continuity: abrupt changes can cause unusually large changes in the measured flow. The authors report that this additional optimization reduces some sharp temporal transitions.

This is a separate idea from extended DMD. Extended DMD uses the teacher's score to guide sampled windows; GRPO uses a motion-related reward to encourage smoother generated videos. A simple smoothness signal is not a complete measure of motion quality: a video can be smooth because it barely moves. [Self-Forcing++, Section 3.3 and Section 4.3.5]




## Python-Style Pseudocode

The pseudocode below illustrates the **long-rollout and sampled-window DMD** idea. It is based on the paper's algorithm, not a line-by-line translation of the authors' implementation. In particular, it shows one possible way to prepare a trainable student prediction and then noise it for DMD; the paper does not specify every detail of that sequence. The optional GRPO stage is omitted here. [Self-Forcing++, Algorithm 1](https://openreview.net/pdf/be9702647edc6df7220cf7f97f1d6672f0e68fa4.pdf)

```python
# Already initialized:
# student: few-step autoregressive video generator
# teacher: frozen diffusion model estimating the real score
# fake_model: separately trained model estimating the student's score
# scheduler: adds noise to a clean latent at a chosen noise level
# student_optimizer: updates the student

for prompt in training_prompts:
    # N = number of chunks in the long rollout
    # K = number of chunks in the short training window
    start = random_integer(0, N - K)
    cache = empty_rolling_kv_cache()
    rollout = []

    # 1. Generate a long video using the student's own history.
    #    Save the cache at the beginning of the chosen window.
    with torch.no_grad():
        for chunk_index in range(N):
            if chunk_index == start:
                window_history = copy_cache(cache)

            noise = torch.randn(chunk_shape)

            # Generate one chunk with the student's few denoising steps.
            clean_chunk = student.generate_chunk(noise, cache, prompt)
            rollout.append(clean_chunk)

            # Write this chunk to the cache; evict old entries if needed.
            cache = student.update_rolling_cache(clean_chunk, cache, prompt)

    # A short segment may come from far into the generated video.
    window = torch.cat(rollout[start:start + K], dim=frame_dimension)

    # 2. Illustrative backward noise initialization.
    #    Prepare a noisy input related to the selected rollout segment.
    t_input = sample_noise_level()
    input_noise = torch.randn_like(window)
    noisy_input = scheduler.add_noise(window, input_noise, t_input)

    # Predict a clean window with gradients enabled.
    # The student uses the saved history and causal attention within the window.
    student_window = student.predict_window(
        noisy_input, t_input, window_history, prompt
    )

    # 3. Obtain the DMD direction for the student prediction.
    with torch.no_grad():
        t_dmd = sample_noise_level()
        dmd_noise = torch.randn_like(student_window)
        noisy_for_dmd = scheduler.add_noise(student_window, dmd_noise, t_dmd)

        # Both scores are evaluated on the same noisy student window.
        real_score = teacher.score(noisy_for_dmd, t_dmd, prompt)
        fake_score = fake_model.score(noisy_for_dmd, t_dmd, prompt)
        dmd_direction = fake_score - real_score

    # 4. Use a detached MSE target to apply the DMD direction.
    target = (student_window - dmd_direction).detach()
    loss = 0.5 * ((student_window - target) ** 2).mean()

    student_optimizer.zero_grad()
    loss.backward()
    student_optimizer.step()

    # The fake model also needs its own separate denoising updates on
    # student-generated samples so it can track the student distribution.
```

`generate_chunk` represents the student's few-step denoising process; `update_rolling_cache` writes the generated chunk's keys and values while keeping the cache bounded. `predict_window` represents a causal student forward pass over the sampled window. These names describe the operations and are not literal repository APIs.

The example keeps the long rollout and saved history detached, while allowing gradients through the student prediction on the sampled window. The detached MSE target is a surrogate for applying the approximate score-difference direction; its value is not the KL divergence. Actual DMD training also includes timestep-dependent factors and practical weighting.

The two noise additions have different purposes in this illustration: `noisy_input` is fed to the **student**, while `noisy_for_dmd` is fed to the **teacher and fake-score model**. Their exact sequencing in the authors' implementation cannot be confirmed from the paper alone. The [official repository](https://github.com/justincui03/Self-Forcing-Plus-Plus) currently says the code will be released.
