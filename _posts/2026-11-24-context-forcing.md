---
layout: post
categories: [diffusion-models]
title: "15. Context Forcing: Training a Student to Use Long-Term Context"
date: 2026-11-24 12:00:00 +0100
description: "How contextual distillation and slow–fast memory help autoregressive video models use long-term history."
math: true
---

[Context Forcing: Consistent Autoregressive Video Generation with Long Context](https://arxiv.org/abs/2602.06028)

Context Forcing studies a problem that is closely related to Deep Forcing: how can an autoregressive video model remember useful history without letting its KV cache grow indefinitely?

The two papers approach the problem differently. Deep Forcing is a training-free method that reorganizes the KV cache of an existing model. Context Forcing also manages the cache, but it goes further: it trains a student using a **long-context teacher**, so the student receives guidance about how to continue its own generated history.

The method combines two ideas:

1. **Contextual distribution matching:** train the student to generate continuations that match a teacher conditioned on the student's history.
2. **Slow–fast memory:** organize the KV cache into initial, long-term, and recent context, so both teacher and student can use a longer span of history with a bounded cache.

The central question is not only *what should the model remember?* It is also *how can we train the model to use that memory?* [Context Forcing, Introduction](https://arxiv.org/html/2602.06028#S1)

## The Student–Teacher Context Mismatch

Some long-video methods generate a long student rollout but use a short-context teacher to supervise short windows sampled from that rollout. This helps the student encounter its own long-rollout errors, as discussed in Self-Forcing++.

However, the Context Forcing authors point out a limitation: the student may use history from much earlier in the video, while the teacher evaluating a short window cannot see that same long-term context.

Consider a video in which a person places a red bag on a table. Much later, the person returns to the table:

```text
Earlier history: person places a red bag on the table
                                      ↓
Later continuation: person returns to the table
```

A short-window teacher may judge whether the later frames look realistic, but if it cannot see the earlier event, it cannot directly guide the student to remember that particular bag. A teacher with access to relevant earlier context can provide a more informative continuation signal.

This is the paper's **student–teacher context mismatch**. A longer-context teacher reduces that mismatch, although its guidance is still an approximation and depends on what information its memory retains. [Context Forcing, Introduction](https://arxiv.org/html/2602.06028#S1)

## Stage 1: Learn Local Video Generation

The student first learns to generate short videos using local distribution matching. This stage is similar in spirit to DMD in Self Forcing: the student generates short video samples, and a teacher and fake-score model provide a score-difference update.

The goal is to give the student a reasonable starting point for local appearance and motion. This matters because the second stage trains on the student's own history: if that history is severely degraded from the start, it is harder for a teacher to provide reliable continuation guidance.

The paper describes this stage using short windows of approximately one to five seconds. The matching objective and its score-based gradient are approximations in practice; the first stage does not guarantee perfect short videos. [Context Forcing, Section 3.1](https://arxiv.org/html/2602.06028#S3.SS1)

## Stage 2: Contextual Distribution Matching

The second stage asks the student to continue from a longer **self-generated context**. The context teacher sees that history through its own memory and supplies a target for the student's continuation.

Let $$H$$ represent the generated history and $$Y$$ a continuation. Conceptually, the paper aims to match

$$
p_{\theta}(Y\mid H)
\quad\text{to}\quad
p_T(Y\mid H),
$$

where $$p_{\theta}$$ is the student's continuation distribution and $$p_T$$ is the context teacher's. The paper expresses this as

$$
\mathcal L_{\mathrm{CDMD}}
=\mathbb E_{H\sim p_\theta}\left[
D_{\mathrm{KL}}\!\left(p_\theta(Y\mid H)\|p_T(Y\mid H)\right)
\right].
$$

The expectation over $$H$$ matters: the history comes from the **student's own rollout**, rather than always being a clean ground-truth video. This exposes the student to imperfections like those it can encounter during inference.

As in DMD, the method estimates an update using the difference between a fake score and a teacher score. Here both scores are conditioned on the student-generated context. The equation is a conceptual objective; practical training uses estimated scores and an approximate gradient rather than directly evaluating the full KL divergence. [Context Forcing, Section 3.2](https://arxiv.org/html/2602.06028#S3.SS2)

### Why Does the Teacher Need Context?

Using the red-bag example, the teacher does not need to reproduce the student's exact next frame. It provides a distributional signal about plausible continuations **given the earlier history**. The student learns to generate its own continuation while receiving guidance that can account for the retained memory of the bag.

This is stronger than checking whether the new frames look plausible in isolation. But it relies on two things: the memory must actually preserve useful information, and the teacher must be reliable for the student's possibly imperfect history. A long-context teacher does not make those assumptions automatically true.

### Growing the Rollout Gradually

A very long student rollout can drift badly early in training. To avoid starting with the hardest examples, the paper gradually increases the maximum rollout length during the second stage. The student first encounters shorter contexts and then is exposed to longer ones as training progresses.

The method also fully denoises the frames used as context. For the target continuation, it retains random denoising exit steps so that the student receives training signals at different steps. [Context Forcing, Section 3.2](https://arxiv.org/html/2602.06028#S3.SS2)

## Slow–Fast Memory: What Goes into the KV Cache?

Keeping every previous frame in the KV cache would become increasingly expensive. Context Forcing therefore divides a fixed cache into three parts:

```text
[ attention sink ][ slow memory ][ fast memory ]
   initial context    selected past   recent context
```

- **Attention sink:** keeps initial context as an anchor.
- **Slow memory:** retains selected older information over a longer period.
- **Fast memory:** keeps recent context in a rolling first-in, first-out queue.

The teacher and student both use the context management mechanism. Their caches encode the context for their respective models; the idea is to give both access to a compatible retained history, not to assume that every frame of the full video is stored. [Context Forcing, Section 3.3](https://arxiv.org/html/2602.06028#S3.SS3)

## How Does Slow Memory Select History?

Consecutive video frames are often similar. If a scene changes slowly, storing every frame in long-term memory may spend the cache budget on redundant information. The paper proposes **surprisal-based consolidation** to decide when to retain new information.

The basic idea is to compare a key representation with the preceding one:

$$
\operatorname{sim}(k_t,k_{t-1})<\tau
\quad\Longrightarrow\quad
\text{consider the new information for slow memory}.
$$

Here $$k_t$$ and $$k_{t-1}$$ are key representations and $$\tau$$ is a similarity threshold. A lower similarity indicates a larger change and makes the information a candidate for retention. If the slow-memory budget is exceeded, old entries are removed.

For example, a largely unchanged background may produce similar neighboring representations. When a new object enters the scene, the representation may change more, making that moment more likely to be retained. This is an intuition: low similarity is only a proxy for novelty, not a guarantee that the retained content is important or error-free.

In the reported configuration, the cache budget is **21 latent-frame slots**: 3 sink, 12 slow, and 6 fast. The authors use a threshold of 0.95 and evaluate consolidation at two-chunk intervals, retaining the first latent when a chunk is consolidated. These are experimental settings, not universal requirements. [Context Forcing, Sections 3.3 and 4](https://arxiv.org/html/2602.06028#S3.SS3)

## Bounded Positional Encoding

A long video also creates a positional-encoding problem. If temporal positions keep growing far beyond the range used during training, the model must compare keys and queries at unfamiliar position differences.

Context Forcing assigns the retained sink, slow-memory, and fast-memory tokens positions within a **bounded temporal range**. Recent tokens occupy the recent part of the cache, while selected older tokens occupy positions reserved for slow memory.

This is similar in motivation to Deep Forcing's temporal RoPE adjustment: both try to make the position relationships of retained history usable during long generation. The methods use different cache policies. Reassigning a position affects how attention interprets a cached key; it does not change when the original event actually happened. [Context Forcing, Section 3.3](https://arxiv.org/html/2602.06028#S3.SS3)

## A More Robust Context Teacher

The teacher also needs to handle the student's imperfect history. A teacher trained only on clean video prefixes may receive very different context when the student starts to drift.

The paper therefore uses **error-recycling fine-tuning** for the context teacher. During teacher training, some contexts are perturbed using errors collected from earlier model outputs. The teacher learns to predict the intended continuation despite imperfect context.

This is designed to make teacher guidance more robust to student-generated artifacts. It does not ensure that the teacher can recover from every possible failure, especially if the student's history becomes severely degraded. [Context Forcing, Section 3.4](https://arxiv.org/html/2602.06028#S3.SS4)

## Context Forcing vs. Deep Forcing

| | Deep Forcing | Context Forcing |
|---|---|---|
| Main approach | Manage an existing model's cache at inference | Train a student with a context teacher and a managed cache |
| Model training | No additional fine-tuning | Local and contextual distribution-matching stages |
| Long-term cache | Deep Sink plus attention-based selection | Sink plus slow memory selected using key similarity |
| Recent context | Preserved in the cache | Stored in fast memory |
| Position handling | Adjusts temporal RoPE for retained keys | Uses bounded temporal positions for the cache |

Both methods try to retain useful history, but their selection criteria differ. Deep Forcing favors candidate tokens that receive higher relevance scores from recent queries. Context Forcing uses changes in key representations as a proxy for new information when deciding what to consolidate into slow memory.



## Python-Style Pseudocode

The following is a simplified view of **Stage 2: Contextual DMD**. Before this stage, the student has received the short-video training from Stage 1, and the context teacher has been trained to handle long and imperfect histories. The teacher is frozen during student distillation.

```python
# student: few-step autoregressive generator, initialized by Stage 1
# teacher: frozen long-context diffusion teacher
# fake_model: trainable score model for the student distribution
# scheduler: adds diffusion noise to generated latent frames
# student_optimizer and fake_optimizer: separate optimizers

for step, prompt in enumerate(training_prompts):
    # The maximum context length increases as training progresses.
    max_length = rollout_schedule(step)
    context_length = random_integer(min_length, max_length)

    # 1. Generate the student's own context with full denoising.
    #    The context is treated as fixed for this simplified student update.
    with torch.no_grad():
        context = student.generate_context(
            prompt, context_length, fully_denoise=True
        )

    # 2. Generate a continuation from that context.
    #    The selected exit step allows training at different denoising steps.
    exit_step = random_choice(denoising_steps)
    continuation = student.generate_continuation(
        prompt, context, exit_step, with_grad=True
    )

    # 3. Add noise to the continuation and ask both score models
    #    to evaluate it using the same student-generated context.
    with torch.no_grad():
        t = sample_dmd_timestep()
        noise = torch.randn_like(continuation)
        noisy_continuation = scheduler.add_noise(continuation, noise, t)

        real_score = teacher.score(noisy_continuation, t, context, prompt)
        fake_score = fake_model.score(noisy_continuation, t, context, prompt)
        dmd_direction = fake_score - real_score

    # 4. Apply the approximate score-difference update to the student.
    target = (continuation - dmd_direction).detach()
    student_loss = 0.5 * ((continuation - target) ** 2).mean()

    student_optimizer.zero_grad()
    student_loss.backward()
    student_optimizer.step()

    # 5. Train the fake model separately to predict noise added to
    #    student-generated continuations. The student is fixed here.
    with torch.no_grad():
        fake_sample = continuation.detach()
        t_fake = sample_dmd_timestep()
        fake_noise = torch.randn_like(fake_sample)
        noisy_fake = scheduler.add_noise(fake_sample, fake_noise, t_fake)

    predicted_noise = fake_model.predict_noise(
        noisy_fake, t_fake, context, prompt
    )
    fake_loss = ((predicted_noise - fake_noise) ** 2).mean()

    fake_optimizer.zero_grad()
    fake_loss.backward()
    fake_optimizer.step()
```

The method names describe operations rather than literal repository APIs. `generate_context` performs the student's autoregressive rollout with the cache policy below; `generate_continuation` produces trainable target predictions conditioned on that history. The teacher and fake model build the context representations appropriate to their own parameters.

The DMD update is schematic: real implementations include timestep-dependent scaling and practical normalization. The detached MSE target applies an approximate score-difference direction; its value is **not** the KL divergence. The fake-model loss illustrates a common noise-prediction parameterization; another implementation may use an equivalent score, clean-prediction, or flow parameterization.

### How the Context Memory Is Updated

During a rollout, the teacher and student each manage their own cache using the same **sink + slow + fast** policy. In simplified form:

```python
# New keys and values are produced when a clean context chunk is encoded.
fast_memory.append(new_chunk_kv)

# Compare the new key representation with the previous one.
# Low similarity is used as a proxy for a meaningful change.
if cosine_similarity(new_key, previous_key) < threshold:
    slow_memory.append(selected_latent_kv)

# Keep both parts within their budgets.
if len(slow_memory) > slow_budget:
    slow_memory.pop(0)  # remove the oldest slow-memory entry
if len(fast_memory) > fast_budget:
    fast_memory.pop(0)  # remove the oldest recent entry

# The initial sink is kept; retained entries receive bounded positions.
attention_cache = sink + slow_memory + fast_memory
```

`selected_latent_kv` denotes the chosen key–value entry from a consolidated chunk. This sketch shows the idea rather than the exact implementation: the paper evaluates consolidation at intervals and retains the first latent of a consolidated chunk in its reported setting. Bounded temporal positions are applied when the retained cache is used; they do not change the actual time of an event. [Context Forcing, Sections 3.2–3.3 and Appendix C](https://arxiv.org/html/2602.06028#S3.SS2)
