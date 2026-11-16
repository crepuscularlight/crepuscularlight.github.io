---
layout: post
categories: [diffusion-models]
title: "9. Causal Forcing: Addressing the Architecture Gap in Self Forcing"
date: 2026-11-16 12:00:00 +0100
description: "How causal ODE initialization addresses the teacher–student architecture gap before asymmetric DMD."
math: true
---

[Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation](https://arxiv.org/abs/2602.02214)

Causal Forcing, by Hongzhou Zhu, Min Zhao, Guande He, and collaborators, revisits a question from CausVid and Self Forcing: **how do we train a causal video generator from a bidirectional diffusion model?**

I found this paper difficult to understand at first because it discusses two different kinds of mismatch. Self Forcing focuses on the difference between training on data context and generating from the student's own context. Causal Forcing studies an earlier stage: the **architecture gap** between a bidirectional teacher and a causal student during ODE initialization.

The authors argue that the student's ODE regression target can depend on future information that the student cannot see. They analyze this through a condition they call **frame-level injectivity**. Their proposed solution is to use a causal teacher for ODE initialization, then apply the asymmetric DMD stage used in Self Forcing. [Causal Forcing, Introduction](https://arxiv.org/html/2602.02214v5#S1)

## The Architecture Gap

A bidirectional video model can use the whole video when predicting a frame: past, present, and future. A causal student predicts a frame or chunk using only its current input and earlier context. This restriction is what allows the student to generate video in order.

In CausVid and Self Forcing, ODE initialization uses trajectories from a **bidirectional** teacher to train a **causal** student. The teacher generates a trajectory from noise to a clean video, and the student learns to predict the teacher's clean output from an intermediate state. The concern is that the teacher's target for a frame may depend on future frames that are visible to the teacher but hidden from the student.

Causal Forcing asks: if the teacher uses information that the student does not have, can the student reproduce the teacher's ODE mapping? [Causal Forcing, Sections 3.1–3.2](https://arxiv.org/html/2602.02214v5#S3.SS2)

## What Does Frame-Level Injectivity Mean Here?

The paper calls its condition *frame-level injectivity*. I find the following distinction helpful:

- **Video level:** the teacher's ODE operates on the entire noisy video. Under the usual ODE uniqueness assumptions, a full state on a trajectory determines its clean endpoint.
- **Frame level:** the causal student sees a smaller input: the current noisy frame or chunk and the context it is allowed to use. That input must contain enough information to determine the teacher's target for the current frame.

These are not the same guarantee. Knowing a full noisy video is more informative than knowing one noisy frame and its past.

Consider a simplified example with two noisy values, $$a$$ and $$b$$. Imagine an invertible teacher mapping that produces two clean targets:

$$
y_1=a+b,\qquad y_2=b.
$$

The full mapping is invertible: from $$(y_1,y_2)$$, we can recover $$b=y_2$$ and $$a=y_1-y_2$$. But the first output $$y_1$$ is not determined by $$a$$ alone: it also depends on $$b$$. If a causal student predicting the first output can see $$a$$ but not $$b$$, the teacher may provide different targets for the same available input.

This is only an illustration, not the paper's video model. In an actual video, $$b$$ represents information in future frames that can influence the bidirectional teacher's prediction but is unavailable to the causal student.

The paper states its frame-level condition as the same noisy frame at the same timestep having the same corresponding clean target. For understanding a **conditional** causal model, it is important to keep its available history fixed too: different histories can legitimately lead to different outputs. The practical issue is ambiguity after conditioning on *everything the student actually sees*. The paper's analysis relates this ambiguity to the bidirectional teacher's dependence on information outside the student's input. [Causal Forcing, Section 3.2](https://arxiv.org/html/2602.02214v5#S3.SS2)

## Why Can ODE Regression Become Blurry?

Suppose the current noisy frame is $$x_t^i$$, and the causal student's available history is $$h^{<i}$$. During ODE initialization, the student is trained with a mean-squared regression loss against the teacher's clean target $$x_0^i$$:

$$
\mathcal L_{\mathrm{ODE}}(\theta)
=\mathbb E\left[
\left\|G_\theta(x_t^i,h^{<i},t)-x_0^i\right\|_2^2
\right].
$$

If different teacher trajectories can produce different clean targets for the same information available to the student, a deterministic student cannot reproduce all of them. For squared-error regression, the optimal prediction is the conditional mean:

$$
G^*(x_t^i,h^{<i},t)
=\mathbb E\left[x_0^i\mid x_t^i,h^{<i},t\right].
$$

For intuition, imagine that a future frame tells the teacher whether an object should end up slightly to the left or right. If that information is unavailable to the student, averaging the possible targets can produce a less sharp result. The paper connects this conditional-mean behavior to blurry or inconsistent outputs after ODE initialization. This is a statement about the regression problem and its assumptions; it does not mean every causal video model must be blurry. [Causal Forcing, Section 3.2](https://arxiv.org/html/2602.02214v5#S3.SS2)

## The Three Stages of Causal Forcing

The method changes the teacher used for **ODE initialization**, while retaining the strengths of a bidirectional teacher for the later distribution-matching stage.

### 1. Train a Causal Diffusion Teacher

The authors first adapt a pretrained bidirectional diffusion model into a many-step autoregressive diffusion model. They use **teacher forcing**: when training the current frame, the model is conditioned on clean earlier frames from the training sequence.

The paper compares this with a Diffusion Forcing setup in which earlier frames are noised, and reports better results with teacher forcing for its causal teacher. Clean context in this stage should not be confused with perfect inference alignment: at inference, the model's earlier frames are generated, so exposure bias can still matter. The later self-rollout stage addresses that different issue. [Causal Forcing, Section 3.3](https://arxiv.org/html/2602.02214v5#S3.SS3)

### 2. Initialize the Student with Causal ODE Trajectories

The authors then use the causal diffusion model as the ODE teacher. For each current frame or chunk, they:

1. Take a clean history from a training sequence.
2. Start the current frame from Gaussian noise and run the causal teacher's ODE solver, conditioned on that history.
3. Save intermediate noisy states and the teacher-generated clean endpoint.
4. Train the causal student to predict that endpoint from an intermediate state and the **same clean history**.

The regression objective can be written as

$$
\mathcal L_{\mathrm{causal\text{-}ODE}}(\theta)
=\mathbb E\left[
\left\|G_\theta(x_t^i,h_{\mathrm{clean}}^{<i},t)
-x_0^i\right\|_2^2
\right].
$$

Here, $$x_t^i$$ and $$x_0^i$$ belong to the causal teacher's trajectory for the current frame, while $$h_{\mathrm{clean}}^{<i}$$ is its conditioning history. The target $$x_0^i$$ is generated by the teacher; it is not necessarily the next frame from the history sequence.

The intended benefit is that the causal teacher's target is compatible with the information supplied to the causal student. For a fixed history and timestep, the teacher's conditional ODE maps the current noisy state to an endpoint without looking at future frames. This relies on the paper's ODE and modeling assumptions; changing attention masks alone does not prove that every practical model will learn the mapping perfectly. [Causal Forcing, Section 3.3](https://arxiv.org/html/2602.02214v5#S3.SS3)

### 3. Apply Asymmetric DMD

After causal ODE initialization, Causal Forcing follows Self Forcing's DMD procedure. The causal student generates its own video context through self-rollout, while a pretrained **bidirectional** diffusion model supplies the real-score estimate. A fake-score model tracks the student's generated distribution.

This initially seemed contradictory to me: why use a causal teacher in one stage and a bidirectional teacher in the next? The objectives are different:

- **ODE initialization** regresses a particular teacher trajectory's endpoint, so the paper argues that the target must be compatible with the student's available information.
- **DMD training** compares distributions through score estimates. It does not require the causal student to reproduce the bidirectional teacher's particular output for each input noise sample.

Thus, “causal” in Causal Forcing does not mean the bidirectional teacher disappears from the whole pipeline. The causal teacher is used for ODE initialization; the later DMD stage remains asymmetric. [Causal Forcing, Sections 3.3 and 5](https://arxiv.org/html/2602.02214v5#S3.SS3)

## How Does This Relate to Self Forcing?

Self Forcing and Causal Forcing address related but different questions:

- **Self Forcing:** During training, does the student see the kind of self-generated history it will encounter at inference?
- **Causal Forcing:** During ODE initialization, can the causal student reproduce targets from a teacher that has access to future information?

Causal Forcing keeps the Self Forcing-style DMD stage, but changes how the student is initialized before that stage. The authors report better motion and visual-quality scores than their Self Forcing baseline under their evaluation setup. They also note that long-video extrapolation remains a separate challenge: the models are trained on short clips, so generating much longer videos can still lead to degradation. [Causal Forcing, Sections 4–5](https://arxiv.org/html/2602.02214v5#S4)

For me, the main insight is that **a teacher's target is only useful if the student has enough information to learn it**. Causal Forcing applies that idea to ODE trajectory regression, then uses DMD and self-rollout to refine the initialized student as a video generator.

## Causal Forcing Pseudocode

The following Python-style pseudocode shows the three stages. It focuses on **which model is used and what it learns**; data loading, model construction, and optimizer setup are left out. `history` contains earlier clean chunks, `current` is the current clean chunk, and `prompt` is the text condition. Video tensors are assumed to have shape `[batch, frames, channels, height, width]`.

### 1. Train the Causal Diffusion Teacher

In this example, the causal teacher predicts a flow velocity. The current chunk is noised, while the history stays clean:

```python
import torch
import torch.nn.functional as F

for history, current, prompt in teacher_training_batches:
    t = torch.rand(()).item()  # A noise level between 0 and 1.
    noise = torch.randn_like(current)
    noisy = (1 - t) * current + t * noise

    # The history comes from the training sequence: teacher forcing.
    predicted_flow = causal_teacher(noisy, history, t, prompt)
    target_flow = noise - current
    loss_teacher = F.mse_loss(predicted_flow, target_flow)

    optimizer_teacher.zero_grad()
    loss_teacher.backward()
    optimizer_teacher.step()
```

The model learns to denoise the current chunk conditioned on earlier clean chunks. The loss does not ask it to see future chunks.

### 2. Generate Causal ODE Pairs and Train the Student

Now freeze the causal teacher. Starting from noise, follow its ODE while keeping the same clean history. Each intermediate state and the final teacher-generated endpoint form a regression pair:

```python
causal_teacher.requires_grad_(False)
ode_pairs = []

# Example integration schedule from noise (t=1) to clean (t=0).
times = torch.linspace(1.0, 0.0, 51).tolist()

for history, current, prompt in ode_data_batches:
    with torch.no_grad():
        x = torch.randn_like(current)
        trajectory = []

        for t, next_t in zip(times[:-1], times[1:]):
            trajectory.append((x.clone(), t))
            velocity = causal_teacher(x, history, t, prompt)
            x = x + (next_t - t) * velocity  # One Euler ODE step.

        teacher_endpoint = x.detach()
        for noisy_state, t in trajectory:
            ode_pairs.append((history, noisy_state, t, teacher_endpoint, prompt))

for history, noisy_state, t, teacher_endpoint, prompt in ode_pair_batches:
    prediction = student(noisy_state, history, t, prompt)
    loss_ode = F.mse_loss(prediction, teacher_endpoint)

    optimizer_student.zero_grad()
    loss_ode.backward()
    optimizer_student.step()
```

`ode_pair_batches` samples batches from `ode_pairs`. The teacher and student receive the **same history** for each pair, and the target is the causal teacher's endpoint. Euler integration is shown to make the trajectory explicit; the actual solver, stored timesteps, batching, and model parameterization depend on the implementation.

### 3. Refine the Student with Self-Rollout and DMD

Finally, the student generates a video using its **own earlier predictions** as history. The DMD teacher is now bidirectional. To keep the rollout manageable, this example stops gradients through the earlier denoising calls and generated history, as in the Self Forcing idea:

```python
# student_steps is a decreasing list of noise levels, for example
# [1.0, 0.75, 0.5, 0.25]. The list contains no clean endpoint (0).
for prompt in prompt_batches:
    generated_chunks = []
    stop = torch.randint(len(student_steps), ()).item()

    for _ in range(num_chunks):
        history = (torch.cat(generated_chunks, dim=1).detach()
                   if generated_chunks else None)
        x = torch.randn(chunk_shape, device=device)

        # Earlier denoising calls build the input without keeping gradients.
        with torch.no_grad():
            for j in range(stop):
                clean = student(x, history, student_steps[j], prompt)
                next_t = student_steps[j + 1]
                x = (1 - next_t) * clean + next_t * torch.randn_like(clean)

        # Keep a gradient through the selected call for this chunk.
        clean = student(x, history, student_steps[stop], prompt)
        generated_chunks.append(clean)

    video = torch.cat(generated_chunks, dim=1)

    # Compare the bidirectional teacher and fake model on the same video.
    t = 0.02 + 0.96 * torch.rand(()).item()
    noise = torch.randn_like(video)
    with torch.no_grad():
        noisy_video = (1 - t) * video + t * noise
        real_clean = bidirectional_teacher.predict_clean(noisy_video, t, prompt)
        fake_clean = fake_model.predict_clean(noisy_video, t, prompt)
        scale = (video - real_clean).abs().mean(
            dim=(1, 2, 3, 4), keepdim=True
        ).clamp_min(1e-6)
        grad = (fake_clean - real_clean) / scale

    target = (video - grad).detach()
    loss_dmd = 0.5 * F.mse_loss(video, target)
    optimizer_student.zero_grad()
    loss_dmd.backward()
    optimizer_student.step()

    # Train the fake model to track detached student-generated videos.
    fake_noise = torch.randn_like(video)
    noisy_fake = (1 - t) * video.detach() + t * fake_noise
    predicted_flow = fake_model.predict_flow(noisy_fake, t, prompt)
    loss_fake = F.mse_loss(predicted_flow, fake_noise - video.detach())
    optimizer_fake.zero_grad()
    loss_fake.backward()
    optimizer_fake.step()
```

The important distinction is the two teachers: `causal_teacher` produces paired ODE targets in Stage 2; `bidirectional_teacher` supplies a target-distribution estimate for DMD in Stage 3. `fake_model` is trained separately to follow the student's distribution. The detached-target MSE is a gradient surrogate; its value is not the KL divergence.

This is an educational outline, not runnable repository code. It omits classifier-free guidance, timestep weighting, KV caching, and the exact model wrappers and update schedule. The paper's implementation uses chunks of latent frames, and its exact loss parameterization and solver should be taken from the implementation. [Causal Forcing, Section 3.3](https://arxiv.org/html/2602.02214v5#S3.SS3)
