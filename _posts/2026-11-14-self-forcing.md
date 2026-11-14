---
layout: post
categories: [diffusion-models]
title: "7. Self Forcing: Bridging the Train–Test Gap in Autoregressive Video Diffusion"
date: 2026-11-14 12:00:00 +0100
description: "How self-rollout, video-level distribution matching, and KV caching improve autoregressive video generation."
math: true
---

[Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009)

While working on RoboEnvision, I noticed that autoregressive video generation can accumulate artifacts as it produces frames sequentially. I found this difficult to solve because an error in an early frame can affect everything generated after it. Self Forcing gives me a useful way to understand the problem: it focuses on the difference between the context a model sees during training and the context it sees during inference.

The paper, by Xun Huang and collaborators, builds on ideas from autoregressive video diffusion and distribution matching. Its central idea is to let the model generate its **own context during training**, then supervise the resulting video as a whole. I find the work especially interesting in relation to CausVid and the later forcing-based methods it helped inspire.

## Preliminary: Autoregressive Video Diffusion

An autoregressive video model generates a sequence in order. Given earlier frames, it predicts the next frame or chunk. We can describe the distribution of a video with $$N$$ frames as

$$
p_\theta(x^{1:N})=\prod_{i=1}^{N}p_\theta(x^i\mid x^{<i}).
$$

Here, $$x^{<i}$$ represents the earlier frames. In an autoregressive diffusion model, each conditional generation uses a diffusion process: the model starts from noise and denoises the current frame or chunk while conditioning on what came before. There are therefore two different kinds of steps: **video steps**, which move forward through frames or chunks, and **denoising steps**, which refine the current frame or chunk.

The paper studies both frame-wise and chunk-wise generation. For simplicity, I will use “chunk” below; the same intuition also applies when a chunk contains only one latent frame. [Self Forcing, Section 3.1](https://arxiv.org/html/2506.08009v2#S3.SS1)

## The Training–Inference Mismatch

In teacher forcing, the model learns to generate the next part of a video using earlier **ground-truth** frames as context. During inference, however, ground-truth future video is unavailable: the model must use its own generated frames. If those frames contain small errors, later predictions are conditioned on inputs that differ from the clean context seen during training. This mismatch is called **exposure bias**.

Diffusion Forcing expands the training setup by giving different frames or chunks different noise levels. The context can therefore contain noisy versions of earlier data frames. This exposes the model to a wider range of noise, but those context frames still come from the data distribution rather than from the model's own autoregressive rollout. Added noise is not necessarily the same as the model's generation errors.

For example, suppose a generated object's shape is slightly distorted in an early chunk. At inference, the next chunk must continue from that distorted object. Training only on real or noised real context may not prepare the model for the same type of mistake. Over many chunks, such errors can accumulate. [Self Forcing, Section 3.1](https://arxiv.org/html/2506.08009v2#S3.SS1)

## Self Forcing: Training on the Model's Own Rollout

Self Forcing changes how the training video is produced:

1. Generate the first chunk from noise with the current student.
2. Generate the next chunk conditioned on the student's previous output.
3. Repeat this process to form a video, using the same autoregressive idea as inference.
4. Apply a video-level distribution-matching objective to the generated sequence and update the student.

This process is called **self-rollout**. The model encounters its own imperfect outputs as context during training, giving it a chance to learn how to continue from them. Self Forcing reduces a source of exposure bias; it does not guarantee that errors disappear, especially for videos much longer than those used during training.

The method uses a key–value (KV) cache during the rollout. The cache stores attention information from previous generated chunks so the model can reuse that context when generating the next one. Caching makes the sequential process more efficient; the core change in the training distribution comes from using the student's own outputs as context. [Self Forcing, Section 3.2](https://arxiv.org/html/2506.08009v2#S3.SS2)

## A Video-Level Distribution-Matching Loss

Once the student has generated a sequence, how do we train it without a target video paired with the exact same noise? One choice is DMD, which compares distributions instead of requiring a frame-by-frame match to a particular teacher output.

As in the DMD and CausVid posts, the DMD version uses two score estimates:

- A pretrained **real-score model** describes the target video distribution.
- A **fake-score model** learns the distribution of videos currently produced by the student.

The difference between these scores provides an approximate update for the student. Both scores are evaluated on a noised version of the generated video. The target score can come from a bidirectional video model during training, while the student itself remains causal during generation.

For a generated video $$\hat{x}_0$$ and a noise level $$t$$, we add noise to the whole video:

$$
\hat{x}_t=\alpha_t\hat{x}_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Let $$p_{\theta,t}$$ be the distribution of noisy student-generated videos and $$p_{\mathrm{real},t}$$ the noisy target distribution. The DMD objective compares them using the reverse KL divergence:

$$
\mathcal L_{\mathrm{DMD}}(\theta)
=\mathbb E_t\left[
D_{\mathrm{KL}}\!\left(p_{\theta,t}\,\|\,p_{\mathrm{real},t}\right)
\right].
$$

We do not need to evaluate the KL directly. For a fixed $$t$$, its gradient can be written in terms of the two scores. With $$J_\theta=\partial\hat{x}_0/\partial\theta$$,

$$
\nabla_\theta D_{\mathrm{KL}}
=\mathbb E\left[
\alpha_t J_\theta^\top
\left(s_{\mathrm{fake}}(\hat{x}_t,t)
-s_{\mathrm{real}}(\hat{x}_t,t)\right)
\right].
$$

A score is the gradient of a noisy distribution's log density with respect to the video. During gradient descent, the sign reverses: the teacher's real score guides samples toward the target distribution, while the fake score accounts for what the student already produces. The expectation averages over student rollouts and added noise, and the update also averages over timesteps. In practice both scores are estimated, and implementations may weight the update by noise level. Self Forcing additionally truncates the gradient through the rollout, as explained below.

The important difference from a frame-wise denoising loss is that this supervision is applied to the **generated video sequence as a whole**. The goal is to improve the distribution of complete student rollouts, including their temporal behavior. This does not mean the teacher provides a separate ground-truth correction for each generated frame.

DMD is one option, not a requirement of Self Forcing. The paper also studies Score Identity Distillation (SiD) and GAN objectives. Its DMD and SiD implementations can be trained without a video dataset during the Self Forcing stage; the GAN setup in the paper uses a dataset of teacher-generated videos. [Self Forcing, Section 3.3](https://arxiv.org/html/2506.08009v2#S3.SS3)

## Making the Rollout Practical

Generating multiple chunks sequentially and backpropagating through every denoising step would be expensive. Self Forcing makes this feasible in several ways.

First, it starts with a **few-step model**. Following CausVid's approach, the paper initializes a causal student using teacher-generated ODE solution pairs, then applies Self Forcing as a post-training stage. Its experiments use four denoising steps; the few-step generator makes repeated training rollouts more affordable. [Self Forcing, Section 4](https://arxiv.org/html/2506.08009v2#S4)

Second, the paper uses **stochastic gradient truncation**. Instead of backpropagating through the complete denoising chain, it samples a stopping step during training and keeps gradients through that final selected denoising call for each chunk. Earlier denoising calls run without gradient tracking. Sampling different stopping steps across iterations allows the model to receive supervision at different points in the few-step process.

The KV embeddings used as context are also detached, so gradients from a later chunk do not backpropagate through earlier chunks via the cache. The forward rollout still uses the student's previous outputs as context, but the training gradient is truncated. Thus, “matching inference” describes how context is generated; training does not backpropagate through the entire inference trajectory. [Self Forcing, Section 3.2](https://arxiv.org/html/2506.08009v2#S3.SS2)

## Long Videos and the Rolling KV Cache

A cache that grows with every generated chunk would eventually become expensive. For longer videos, Self Forcing uses a **rolling KV cache**: it keeps a fixed amount of recent context and removes the oldest entries as new chunks are generated. This allows the model to reuse cached attention information without repeatedly rebuilding the whole context window.

There is a subtle issue when old entries are removed. In the paper's video representation, the initial latent frame has different properties from later latent frames. If the model always sees that first frame during training, generation can develop artifacts when the rolling cache eventually drops it. The authors therefore restrict attention during training so that later chunks sometimes do not see the first chunk, helping the model handle that situation at inference. [Self Forcing, Section 3.4](https://arxiv.org/html/2506.08009v2#S3.SS4)

## How Is This Different from CausVid?

CausVid and Self Forcing both use causal students and can use DMD with a bidirectional teacher. The main difference I want to remember is **how the student's training inputs are produced**.

CausVid's DMD training uses a Diffusion Forcing-style setup: training video chunks are noised at different timesteps and passed through the causal student. At inference, however, later chunks are conditioned on the student's own earlier generated chunks. The Self Forcing authors point out that this means the videos used for distribution matching during training do not necessarily follow the student's actual autoregressive inference distribution.

Self Forcing instead generates its training sequence through a student rollout. It then applies the distribution-matching objective to that generated sequence. The two approaches therefore share the idea of distribution matching, but Self Forcing changes the **distribution of student samples being matched**. Training also uses stochastic stopping and truncated gradients, so it is an efficient approximation rather than a full backpropagation through inference.

In the paper's comparison using the same Wan-1.3B base model, the chunk-wise Self Forcing model achieves a VBench total score of 84.31 versus 81.20 for the compared CausVid implementation; both report 17.0 FPS in that setup. These results are specific to the models, implementation, and evaluation used in the paper. The authors also observe that quality can still degrade on videos substantially longer than the training context. [Self Forcing, Sections 4 and 5](https://arxiv.org/html/2506.08009v2#S4)

For me, the central lesson is that a loss alone is not the whole story. For an autoregressive model, we also need to ask **which context produces the samples that the loss supervises**. Self Forcing makes the model practice on its own generated context, addressing the source of error accumulation that I encountered in video generation.

## Self Forcing Pseudocode

The following Python-style pseudocode illustrates **Self Forcing with DMD**. It shows the rollout and losses explicitly, while leaving the model architectures and KV-cache implementation abstract. It assumes:

- `student(x, t, cache, prompt)` predicts a clean latent chunk. `student.update_cache(clean, cache, prompt)` adds that chunk's attention keys and values to the cache.
- `teacher(noisy_video, t, prompt)` and `fake_model(noisy_video, t, prompt)` predict the noise in a whole video. The teacher is frozen; the fake model is trained on student outputs.
- `steps` is a decreasing list of student denoising timesteps, and `alpha[t]`, `sigma[t]` define the noise schedule. Video tensors have shape `[batch, channels, frames, height, width]`.

```python
import torch
import torch.nn.functional as F

teacher.requires_grad_(False)
teacher.eval()

for prompt in prompt_loader:
    cache = None
    chunks = []

    # Choose the last denoising step used for this training rollout.
    stop = torch.randint(len(steps), ()).item()

    for _ in range(num_chunks):
        # Start this chunk from noise, conditioned on previously cached chunks.
        x = torch.randn(chunk_shape, device=device)

        # Earlier denoising calls have no gradient.
        with torch.no_grad():
            for j in range(stop):
                clean = student(x, steps[j], cache, prompt)
                next_t = steps[j + 1]
                x = alpha[next_t] * clean + sigma[next_t] * torch.randn_like(clean)

        # Only the selected final call keeps a gradient for this chunk.
        clean = student(x, steps[stop], cache, prompt)
        chunks.append(clean)

        # Later chunks use this generated context, but do not backpropagate
        # through the cache into earlier chunks.
        with torch.no_grad():
            cache = student.update_cache(clean.detach(), cache, prompt)

    video = torch.cat(chunks, dim=2)

    # DMD: compare the teacher and fake model on the same noised video.
    t = torch.randint(1, len(alpha) - 1, ()).item()
    with torch.no_grad():
        noise = torch.randn_like(video)
        noisy_video = alpha[t] * video + sigma[t] * noise
        eps_real = teacher(noisy_video, t, prompt)
        eps_fake = fake_model(noisy_video, t, prompt)
        grad_video = (alpha[t] / sigma[t]) * (eps_real - eps_fake)

    # Surrogate whose gradient supplies the DMD direction to the student.
    loss_dmd = (video * grad_video).flatten(1).sum(1).mean()
    optimizer_student.zero_grad()
    loss_dmd.backward()
    optimizer_student.step()

    # Train the fake model to predict noise added to detached student videos.
    t_fake = torch.randint(1, len(alpha) - 1, ()).item()
    noise = torch.randn_like(video)
    noisy_fake = alpha[t_fake] * video.detach() + sigma[t_fake] * noise
    eps_pred = fake_model(noisy_fake, t_fake, prompt)
    loss_fake = F.mse_loss(eps_pred, noise)
    optimizer_fake.zero_grad()
    loss_fake.backward()
    optimizer_fake.step()
```

The key step is `student.update_cache(...)`: the next chunk receives context from the student's **generated** output. The cache is updated without gradients, so later chunks do not backpropagate through earlier ones. The randomly selected `stop` is shared across the chunks in this example; changing it across training iterations gives different denoising calls supervision. At inference, the student uses the full chosen denoising schedule.

`loss_dmd` is a surrogate: its value is not a KL divergence, but its gradient passes the detached score-difference direction through the generated video. `loss_fake` separately trains the fake model to track the student's output distribution. This example uses one update of each model per loop, omits guidance, timestep weighting, and DMD normalization, and assumes a noise-prediction parameterization with nonzero `alpha[t]` and `sigma[t]` for sampled DMD timesteps. The paper's Wan-based implementation uses a flow-matching parameterization, so its actual model outputs and scaling differ. The code also leaves rolling-cache eviction and local-attention training out of the short training example. [Self Forcing, Algorithm 1 and Appendix A](https://arxiv.org/html/2506.08009v2#S3.SS2)
