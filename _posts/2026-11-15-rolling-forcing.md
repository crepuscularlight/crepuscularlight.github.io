---
layout: post
categories: [diffusion-models]
title: "8. Rolling Forcing: Autoregressive Long Video Diffusion in Real Time"
date: 2026-11-15 12:00:00 +0100
description: "Rolling denoising windows, attention sinks, and training for long autoregressive video generation."
math: true
---

[Rolling Forcing](https://arxiv.org/abs/2509.25161)

After reading Self Forcing, I became interested in how autoregressive video models can maintain quality over longer generations. Self Forcing reduces the gap between training and inference by letting the student train on its own generated context. However, long videos can still drift: small errors in color, appearance, or motion can accumulate as generation continues.

Rolling Forcing, by Kunhao Liu and collaborators, addresses this problem while keeping streaming generation fast. Its main ideas are a **rolling denoising window**, an **attention sink** that retains early context, and an efficient post-training procedure that builds on Self Forcing. The paper reports generating multi-minute videos at approximately 16 FPS on a single GPU in its evaluation setup. [Rolling Forcing, Introduction](https://arxiv.org/html/2509.25161#S1)

Before reading this paper, I was already interested in two ways to improve long-video consistency. Diffusion Forcing trains with different noise levels across a sequence, exposing the model to different kinds of noisy context. Another idea is to use keyframes as anchors, which is also related to what I explored in RoboEnvision. Rolling Forcing combines different ideas: it uses a structured noise schedule inside a moving window and preserves the initial generated frames as a form of global context.

## From Self Forcing to a Rolling Window

In Self Forcing's usual chunk-wise generation, the model denoises the current chunk while conditioning on previously generated chunks. The current chunk can use information from its history, but finalized history cannot be revised in response to later chunks. When errors become part of that history, future predictions may inherit them.

Rolling Forcing changes the active denoising process. Instead of working on only one chunk at a time, it keeps a **window of several consecutive chunks** and denoises them jointly. The chunks in this window have different noise levels: earlier chunks are closer to clean, while later ones are noisier. They can attend to each other bidirectionally within the window.

For example, imagine a window containing three chunks:

```text
Current window:  [low noise, medium noise, high noise]
After a pass:    [clean,     low noise,    medium noise]
```

The model emits the clean chunk, shifts the window forward, and adds a new noise sample at the end. Once the process is running, each pass can finalize the oldest chunk while also making progress on the others. The noise levels above are only an illustration; the paper's main setup uses five denoising steps and a window of five chunks.

The key distinction is **where bidirectional attention is allowed**. Chunks inside the active window can refine their predictions using one another. The window still attends causally to the already finalized history: earlier emitted chunks are not regenerated. This limited look-ahead can help reduce the propagation of local errors, although it does not eliminate error accumulation. [Rolling Forcing, Section 3.2](https://arxiv.org/html/2509.25161#S3.SS2)

The rolling schedule reminds me of Diffusion Forcing because different parts of a sequence have different noise levels. Their roles are different, however: Diffusion Forcing is a training framework using independently sampled noise levels, while Rolling Forcing arranges noise levels progressively within a moving window for generation.

### Why Use a Few-Step Model?

In the paper's formulation, the number of denoising stages determines how many chunks occupy the window. If a model needed dozens of denoising steps, jointly processing such a large window would be expensive. Distillation reduces the main setup to five steps, making the rolling window practical on a single GPU. The paper uses chunks of three latent frames, so a window of five chunks contains 15 latent frames. [Rolling Forcing, Sections 3.2 and 4.1](https://arxiv.org/html/2509.25161#S4.SS1)

## Temporal and Global History Context

The active window is only part of the context. The model also uses a KV cache for already generated history, avoiding repeated computation of the same attention keys and values. Rolling Forcing keeps two kinds of history:

- **Temporal context:** recent generated frames, which help the next window continue smoothly.
- **Global context:** the initial generated frames, whose cached keys and values act as a persistent reference for the longer video.

The authors call the second part an **attention sink**, drawing on the idea of retaining initial tokens in streaming language models. In this video method, the retained initial frames can help stabilize properties such as appearance and color over time. This is related to my interest in keyframe anchors, but these are the video's own initial generated frames; the method does not require a separately generated keyframe plan. [Rolling Forcing, Section 3.3](https://arxiv.org/html/2509.25161#S3.SS3)

I initially wondered whether the attention sink was related to differential attention. In this paper, the important mechanism is simpler: **keep selected early frames in the KV cache** so later windows can still attend to them. The method does not rely on differential attention.

Keeping early frames introduces a position problem. As the video grows, the distance between those frames and the current window becomes much larger than the model saw during training. Rolling Forcing dynamically adjusts their rotary position embeddings (RoPE), assigning the global context an effective position just before the recent context. This keeps their relative position within a useful range while allowing them to remain a long-term reference.

The anchor has a limitation: once middle-of-video frames leave the recent cache, they are not retained as explicit global memory. Keeping the beginning and the recent past therefore does not guarantee that every event in a long video can be remembered.

## Rolling Forcing Post-Training

Like Self Forcing, Rolling Forcing uses **self-generated history during training**. The student rolls its own video forward, so the context contains the kinds of imperfections it may see at inference. A DMD objective then supervises the generated video using a pretrained teacher and a fake-score model.

The rolling window creates an efficiency problem, however. Consecutive windows overlap heavily. For example, windows `[1, 2, 3]`, `[2, 3, 4]`, and `[3, 4, 5]` repeat many of the same chunks. Keeping gradients through every full-window prediction would require substantial memory.

The paper instead selects a subset of **non-overlapping windows for gradient computation**. The model still rolls the window forward to generate its history, but only selected windows retain gradients and contribute clean predictions to the video used for DMD supervision. With a randomly chosen offset, different windows can be selected across training iterations. This reduces the number of forward passes that need gradients while keeping the training context self-generated. [Rolling Forcing, Section 3.4](https://arxiv.org/html/2509.25161#S3.SS4)

There is another subtlety: a selected rolling window contains chunks predicted from different noise levels, so their predicted clean outputs may have different levels of quality. The authors find that training only this way can lead to unnatural camera movement. Their solution is **mixed training**: alternate between Rolling Forcing and Self Forcing training with equal probability. The Self Forcing objective acts as a regularizer for natural motion. At inference, the model uses the rolling-window procedure.

## How Does It Differ from Self Forcing?

Both methods train with self-generated context and use distribution matching, but they change different parts of the problem:

- **Self Forcing** focuses on exposure bias: train on an autoregressive rollout so the student sees its own generated history.
- **Rolling Forcing** builds on that idea and changes inference as well: jointly denoise several chunks, retain early frames as global context, and train efficiently with selected windows.

The two methods also use the word “rolling” for different things. Self Forcing's **rolling KV cache** manages past context; Rolling Forcing's **rolling denoising window** contains several partially denoised chunks that are jointly processed. Rolling Forcing also uses a KV cache for history.

For me, the most interesting idea is that the model does not have to choose between processing a whole video at once and finalizing each chunk in isolation. A small, jointly denoised window provides local look-ahead, while the cached history provides both recent continuity and an early reference for the rest of the video.

## Rolling-Window Pseudocode

The paper's training algorithm is easier to follow if we first use a window of **three chunks**. Each model call predicts a clean version of all three:

```text
Input window:     [low noise, medium noise, high noise]
Clean prediction: [chunk A,  chunk B,     chunk C]
```

After the call, chunk A is ready to leave the window. Chunks B and C stay and are re-noised at lower noise levels; a new noise sample enters the window. Then the student processes the new window.

### Rolling During Training

The following pseudocode starts after the first window has been prepared. It shows the **Rolling Forcing branch** of training. To keep the example short, it selects the first window and then every third window for gradients.

```python
import torch

# The three timesteps are ordered from low to high noise.
levels = [t_low, t_medium, t_high]
window = [low_noise_chunk, medium_noise_chunk, high_noise_chunk]
selected_predictions = []

for roll in range(number_of_rolls):
    # With a window size of 3, windows 0, 3, 6, ... do not overlap.
    keep_gradient = (roll % 3 == 0)

    with torch.set_grad_enabled(keep_gradient):
        clean = student(window, levels, cache, prompt)
        # clean contains three predictions: [clean A, clean B, clean C].

    if keep_gradient:
        selected_predictions.extend(clean)

    # Prepare the next window without backpropagating through history.
    with torch.no_grad():
        cache = student.update_cache(clean[0], cache, prompt)

        next_A = alpha[t_low] * clean[1] + sigma[t_low] * torch.randn_like(clean[1])
        next_B = alpha[t_medium] * clean[2] + sigma[t_medium] * torch.randn_like(clean[2])
        next_C = torch.randn_like(clean[2])
        window = [next_A, next_B, next_C]

# Combine the selected clean predictions into one video for the DMD loss.
video = torch.cat(selected_predictions, dim=1)
```

`student(...)` is the video model: it predicts a clean version of each noisy input chunk. `student.update_cache(...)` represents the additional model operation that stores attention information from the oldest prediction. These are explanatory names, not the exact functions in the authors' repository. Each chunk is assumed to have shape `[batch, frames, channels, height, width]`.

For example, selecting every third window gives:

```text
Roll 0: [A, B, C]  ← keep all three predictions for DMD
Roll 1: [B, C, D]  ← generate history, without gradients
Roll 2: [C, D, E]  ← generate history, without gradients
Roll 3: [D, E, F]  ← keep all three predictions for DMD
```

The selected windows do not overlap, so their predictions can be joined in time order. The real implementation randomly chooses an offset instead of always starting with roll 0. It also handles the initial growing window and the final shrinking window, and limits gradients to a recent part of longer rollouts. [Official training pipeline](https://github.com/TencentARC/RollingForcing/blob/main/pipeline/rolling_forcing_training.py)

### Updating the Student with DMD

Once we have `video`, we add noise and ask the teacher and fake model to predict its clean version. The repository uses the difference between those two predictions for the DMD update:

```python
import torch.nn.functional as F

with torch.no_grad():
    noise = torch.randn_like(video)
    noisy_video = alpha[t] * video + sigma[t] * noise

    # The teacher prediction includes classifier-free guidance.
    real_clean = teacher.predict_clean(noisy_video, t, prompt)
    fake_clean = fake_model.predict_clean(noisy_video, t, prompt)

    # Normalize the difference for each video.
    scale = (video - real_clean).abs().mean(
        dim=(1, 2, 3, 4), keepdim=True
    ).clamp_min(1e-6)
    grad = (fake_clean - real_clean) / scale

# This surrogate sends the DMD gradient to the selected student predictions.
target = (video - grad).detach()
loss_dmd = 0.5 * F.mse_loss(video, target)
optimizer_student.zero_grad()
loss_dmd.backward()
optimizer_student.step()
```

The MSE is a **gradient surrogate**: its value is not the KL divergence. The fake model is trained separately on detached student videos so it can track the student's changing distribution. In the released configuration, the fake model is updated five times for each student update. The full method also mixes Rolling Forcing training with Self Forcing training. [Official DMD implementation](https://github.com/TencentARC/RollingForcing/blob/main/model/dmd.py)

These examples show the central idea, rather than every detail needed to run the model. In particular, the released model uses a flow-matching parameterization and a scheduler; its cache layout, positional embeddings, guidance, and boundary handling are more involved.

### Inference After Training

At inference, we no longer need the teacher, fake-score model, or DMD loss. Below is the same three-chunk example, now including **initialization**. The student first builds a full window from noise; it then emits one completed chunk on each roll.

```python
import torch

student.eval()
cache = student.init_cache(batch_size, device)  # Start with no video history.
output = []

with torch.no_grad():
    # Warm-up 1: start chunk A at the highest noise level.
    A_high = torch.randn(chunk_shape, device=device)
    clean = student([A_high], [t_high], cache, prompt)
    cache = student.update_cache(clean[0], cache, prompt, position=0)
    A_medium = (alpha[t_medium] * clean[0]
                + sigma[t_medium] * torch.randn_like(clean[0]))

    # Warm-up 2: add a new noisy chunk B and process A and B together.
    B_high = torch.randn_like(A_medium)
    clean = student([A_medium, B_high], [t_medium, t_high], cache, prompt)
    cache = student.update_cache(clean[0], cache, prompt, position=0)
    A_low = alpha[t_low] * clean[0] + sigma[t_low] * torch.randn_like(clean[0])
    B_medium = (alpha[t_medium] * clean[1]
                + sigma[t_medium] * torch.randn_like(clean[1]))

    # The first full window is ready: [low noise, medium noise, high noise].
    C_high = torch.randn_like(B_medium)
    window = [A_low, B_medium, C_high]
    levels = [t_low, t_medium, t_high]

    for _ in range(number_of_output_chunks):
        clean = student(window, levels, cache, prompt)

        # A has now completed all three denoising stages.
        output.append(clean[0])
        cache = student.update_cache(
            clean[0], cache, prompt, position=len(output) - 1
        )

        # B and C advance to lower noise levels; a new chunk enters.
        next_A = alpha[t_low] * clean[1] + sigma[t_low] * torch.randn_like(clean[1])
        next_B = alpha[t_medium] * clean[2] + sigma[t_medium] * torch.randn_like(clean[2])
        next_C = torch.randn_like(clean[2])
        window = [next_A, next_B, next_C]

video_latents = torch.cat(output, dim=1)
video_frames = vae.decode(video_latents)
```

During warm-up, neither A nor B is emitted. The cache calls at `position=0` represent the repository's updates of the oldest chunk's **provisional** prediction at the same position; they do not append multiple finished chunks. The first chunk is emitted after the first full-window pass.

`student.init_cache` and `student.update_cache` are explanatory names for model-specific cache operations, not the repository's literal API. This example shows the growing window and steady streaming, but stops after the requested number of output chunks instead of showing how to drain a finite set of preselected input chunks. The real implementation also handles dynamic RoPE and uses five chunks in its main window. [Official inference pipeline](https://github.com/TencentARC/RollingForcing/blob/main/pipeline/rolling_forcing_inference.py)
