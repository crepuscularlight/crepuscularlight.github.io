---
layout: post
categories: [diffusion-models]
title: "6. Diffusion Forcing: Independent Noise Levels for Sequence Modeling"
date: 2026-11-13 12:00:00 +0100
description: "Independent noise levels for sequence modeling, recurrent training and sampling, and the trade-offs of Diffusion Forcing's original RNN design."
math: true
---

[Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392)

Diffusion Forcing, by Boyuan Chen and collaborators, combines diffusion with autoregressive sequence modeling by assigning independently sampled noise levels to different tokens. The motivation is that adding noise acts as partial masking: a clean token reveals its content, a heavily noised token hides most of it, and intermediate noise levels provide partial information.

Compared with DMD2 and CausVid, I find the original paper less convincing as a video generation system, especially in its recurrent implementation. However, the independent-noise training idea is useful to understand before reading later forcing methods. My reservations concern how the model represents and processes history, rather than the usefulness of allowing different noise levels across a sequence.

## Different Noise Levels for Different Tokens

There are two separate axes: sequence position $$t$$ and diffusion noise level $$k$$. For a clean token $$x_t^0$$, the noisy input is

$$
x_t^{k_t}
=\sqrt{\bar\alpha_{k_t}}\,x_t^0
+\sqrt{1-\bar\alpha_{k_t}}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
$$

Diffusion Forcing samples $$k_t$$ independently across positions. The noise levels do not have to increase with sequence position during training. [Diffusion Forcing, Section 3.2](https://arxiv.org/html/2407.01392#S3.SS2)

For example, the model might receive one nearly clean frame, another with moderate noise, and a third consisting almost entirely of noise. It must learn how much information each input provides. The central change is therefore in the training inputs: the model learns from histories with different degrees of corruption instead of relying exclusively on clean history.

## What This Changes at Sampling Time

The sampling schedule can keep earlier tokens relatively clean while leaving later tokens uncertain, or denoise several tokens together. The authors also demonstrate guidance over future trajectories. [Project explanation](https://www.boyuan.space/diffusion-forcing/)

This helps clarify the motivation behind the method. In a video, the immediate future may already be fairly predictable while events further ahead remain ambiguous. A shared noise level across every frame does not express that distinction directly; separate levels let the sampler choose how quickly to commit to different parts of the sequence.

My original description of full-sequence diffusion as inherently limited to constant-length generation was too broad. A fixed generation window is a common setup, but longer rollouts can also use conditioning and sliding windows. The useful comparison is how naturally each formulation supports those operations, and what computation or additional machinery they require.

## What the RNN Does

The original implementation summarizes history in a recurrent state:

$$
z_t=f_\theta(z_{t-1},x_t^{k_t},k_t),
\qquad
\hat\epsilon_t=\epsilon_\theta(z_{t-1},x_t^{k_t},k_t).
$$

For video, it combines a U-Net with a GRU and a prediction head. The authors favor recurrence for online decision-making and discuss a causal transformer alternative. [Diffusion Forcing, Section 3.2 and Appendix D.2](https://arxiv.org/html/2407.01392#A4.SS2)

The RNN state carries information across sequence positions. It should not be confused with a noisy frame moving through diffusion steps. These are different roles, even though the model must handle both sequence evolution and denoising.

## My Concerns About the Recurrent Design

The recurrent design raises two concerns for me. First, earlier observations must be compressed into a fixed-size state. This is a potential information bottleneck, not proof that the model necessarily forgets those details.

Second, each state update depends on the preceding state. That dependency limits parallel computation across sequence positions.

Recurrence also has a practical advantage: carrying a bounded state can make streaming computation economical. An attention model's ability to revisit stored context comes with storage and computation costs. My preference for attention in this setting needs to be justified by the task and measurements; the use of a GRU alone is not evidence of a weaker paper.

The official repository now distinguishes its temporal-attention implementation on `main` from the original RNN implementation on the `paper` branch. This supports separating the training paradigm from its first architecture, although a later implementation change does not itself establish a controlled performance comparison. [Official repository](https://github.com/buoyancy99/diffusion-forcing)

## Python-Style Pseudocode

This sketch shows recurrent training and a next-token sampling schedule with history stabilization. `model.cell(z_prev, x_noisy, k)` returns a noise prediction and a candidate next hidden state. The sampler keeps the state associated with a small positive noise level as context for the next token. [Diffusion Forcing, Algorithm 1 and Appendix B.4](https://arxiv.org/html/2407.01392#A2.SS4)

```python
class DiffusionForcingModel:
    # Conceptual wiring; the neural-network modules are configured separately.
    def __init__(self, unet, gru, prediction_head, noise_embedding, state_shape):
        self.unet = unet
        self.gru = gru
        self.prediction_head = prediction_head
        self.noise_embedding = noise_embedding
        self.state_shape = state_shape

    def initial_state(self):
        return zeros(self.state_shape)

    def cell(self, z_prev, x_noisy, k):
        # z_prev: summary of earlier tokens; x_noisy: current noisy token.
        # k: current diffusion noise level, embedded for the U-Net.
        features = self.unet(x_noisy, self.noise_embedding(k), z_prev)
        z_next = self.gru(features, z_prev)
        predicted_noise = self.prediction_head(z_next)
        return predicted_noise, z_next


model = DiffusionForcingModel(unet, gru, prediction_head, noise_embedding, state_shape)


def add_noise(x_clean, k, noise):
    return sqrt(alpha_bar[k]) * x_clean + sqrt(1 - alpha_bar[k]) * noise


# Training: independent noise levels across sequence positions.
for clean_sequence in training_loader:
    z = model.initial_state()
    losses = []

    for x_clean in clean_sequence:  # Sequence axis, not diffusion steps.
        k = uniform_integer(1, K)  # Inclusive; sampled independently per token.
        noise = randn_like(x_clean)
        x_noisy = add_noise(x_clean, k, noise)

        predicted_noise, z_next = model.cell(z, x_noisy, k)
        losses.append(mse(predicted_noise, noise))
        z = z_next  # Keep the graph for backpropagation through time.

    optimizer.zero_grad()
    mean(stack(losses)).backward()
    optimizer.step()


# Inference: generate one token at a time, carrying recurrent context.
@no_grad()
def generate(model, num_tokens, z_context, schedule, k_context):
    # schedule descends from K to 0 and includes 0 < k_context < K.
    assert schedule[0] == K and schedule[-1] == 0
    assert 0 < k_context < K and k_context in schedule

    for _ in range(num_tokens):
        x = randn(token_shape)
        next_context = None

        for k, next_k in zip(schedule[:-1], schedule[1:]):
            # Hold past-token context fixed while denoising this token.
            predicted_noise, z_candidate = model.cell(z_context, x, k)

            if k == k_context:
                next_context = z_candidate

            x = reverse_diffusion_step(x, predicted_noise, k, next_k)

        z_context = next_context
        yield x  # The current token is clean, even though context is noisy.
```

`reverse_diffusion_step` abstracts a DDPM or DDIM update. This sketch omits guidance, loss reweighting, batching, and the zero-noise training endpoint. During training, the state advances once per token; during sampling, each denoising call uses the same past context, and only the saved state advances the sequence. The broader method also supports schedules that denoise several future tokens together.
