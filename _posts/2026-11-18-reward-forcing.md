---
layout: post
categories: [diffusion-models]
title: "11. Reward Forcing: Motion-Aware Distribution Matching"
date: 2026-11-18 12:00:00 +0100
description: "How EMA-Sink and reward-weighted distillation improve motion in streaming video generation."
math: true
---

[Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation](https://arxiv.org/abs/2512.04678)

Reward Forcing, by Yunhong Lu and collaborators, studies a problem I find important for long video generation: how can a model maintain consistency over time without making the video almost static?

Methods such as Rolling Forcing retain information from early frames in a KV cache to help control long-term drift. However, the Reward Forcing authors observe that a model can become too dependent on a fixed initial-frame reference. Later frames may repeatedly resemble the beginning of the video, reducing motion.

The paper introduces two ideas:

1. **EMA-Sink:** update the cached global context as older frames leave the local attention window.
2. **Rewarded DMD (Re-DMD):** give generated videos with higher motion-quality rewards greater weight during distribution-matching training.

The two ideas address different parts of the problem: EMA-Sink changes the history available during generation, while Re-DMD changes the training signal. [Reward Forcing, Introduction](https://arxiv.org/html/2512.04678v2#S1)

## Why Can a Fixed Sink Become a Problem?

An autoregressive video model generates new frames or chunks using previously generated context. A sliding attention window keeps recent frames accessible without allowing the cache to grow indefinitely. The trade-off is that older information eventually leaves the window.

An attention sink keeps selected early-frame keys and values as an additional long-term reference. This can help preserve the scene, but the reference remains anchored to the beginning. The paper observes that excessive reliance on these fixed tokens can lead to **frame copying**: the video stays similar to its initial frames instead of developing natural motion.

The aim is therefore not simply to remove the sink. We want the model to retain useful history while allowing its representation of the scene to evolve. [Reward Forcing, Section 3.2](https://arxiv.org/html/2512.04678v2#S3.SS2)

## EMA-Sink: Updating the Global Context

EMA-Sink starts with a fixed-size sink derived from initial frames. As the local window moves forward, the oldest keys and values leave the window. Instead of discarding them completely, the method incorporates them into the sink using an **exponential moving average**:

$$
S_K^{\mathrm{new}}=\alpha S_K^{\mathrm{old}}+(1-\alpha)K_{\mathrm{evicted}},
$$

$$
S_V^{\mathrm{new}}=\alpha S_V^{\mathrm{old}}+(1-\alpha)V_{\mathrm{evicted}}.
$$

Here, $$S_K$$ and $$S_V$$ are the sink's cached keys and values, and $$K_{\mathrm{evicted}}$$ and $$V_{\mathrm{evicted}}$$ come from the frame leaving the local window. The coefficient $$\alpha$$ controls how much of the previous sink is retained at each update.

For example, if $$\alpha=0.9$$, one update combines 90% of the old sink with 10% of the evicted state. This is an illustration of the equation, not a claim about the setting used in every experiment.

The model can then attend to both the **recent local window** and the **updated sink**. Recent frames remain available in detail, while older information is compressed into a fixed-size representation. The sink is a weighted summary of cached features, not a collection from which all past frames can be recovered exactly. Older contributions gradually fade as more updates occur. [Reward Forcing, Section 3.2](https://arxiv.org/html/2512.04678v2#S3.SS2)

## Preliminary: What Is EM?

**Expectation–maximization (EM)** is a method for fitting a model when some useful information is hidden. Rather than guessing the hidden information once and treating the guess as correct, EM works with a probability distribution over the possible answers.

For example, imagine that we have many unlabeled videos and want to group them by their motion. Each video has an unknown group label. We could start with two groups, but at first we do not know which videos belong to which group.

EM alternates between two steps:

1. **E-step (expectation):** use the current model to estimate the probability of each possible group for every video.
2. **M-step (maximization):** update the model to better fit the videos, using those probabilities as weights.

Suppose the current model assigns one video an 80% probability of belonging to group A and a 20% probability of belonging to group B. In the M-step, that video contributes to updating *both* groups, but it has four times as much weight for A as for B. After updating the groups, the model can estimate the probabilities again. It repeats these two steps until the estimates and the model change very little.

### The Objective Behind the Two Steps

Let $$x$$ be an observed sample, $$z$$ be its hidden variable, and $$p_\theta(x,z)$$ be the model's joint probability for both. In the example, $$x$$ is a video and $$z$$ is its unknown group. We want to make the observed videos likely under the model:

$$
\max_\theta \log p_\theta(x)
=\max_\theta \log \sum_z p_\theta(x,z).
$$

We sum over possible values of $$z$$ because we observe the video but not its group. Directly maximizing this expression can be difficult when the hidden variable has many possibilities.

**E-step:** using the current parameters $$\theta_{\mathrm{old}}$$, calculate the posterior probability of each hidden value:

$$
q(z)=p_{\theta_{\mathrm{old}}}(z\mid x).
$$

In the video example, $$q(z)$$ is the model's current probability that the video belongs to group $$z$$.

**M-step:** keep these probabilities fixed and update the model by maximizing the expected log joint probability:

$$
\theta_{\mathrm{new}}
=\arg\max_\theta\;\mathbb E_{z\sim q}\left[\log p_\theta(x,z)\right].
$$

The expectation means that each possible group contributes according to its E-step probability. In other words, the M-step fits the model to the observed video and its possible hidden labels, with more weight on the more likely labels. For a dataset, we add this contribution across all samples.

Exact EM has a useful property: each E-step and M-step does not decrease the observed-data likelihood. In practice, the posterior or the M-step may be approximated, so this guarantee depends on how the approximation is made.

### Why Does EM Work? The ELBO

The E-step and M-step can be understood through the **evidence lower bound (ELBO)**. “Evidence” here means the probability $$p_\theta(x)$$ of the observed data. For any distribution $$q(z)$$ over the hidden variable, we can write

$$
\log p_\theta(x)
=\underbrace{\mathbb E_{z\sim q}\!\left[
\log p_\theta(x,z)-\log q(z)
\right]}_{\mathcal L_{\mathrm{ELBO}}(q,\theta)}
+D_{\mathrm{KL}}\!\left(q(z)\|p_\theta(z\mid x)\right).
$$

A KL divergence is nonnegative, so the ELBO cannot be larger than $$\log p_\theta(x)$$. The KL term is exactly the **gap** between the lower bound and the log likelihood. This gives each EM step a clear purpose:

- **E-step:** set $$q(z)$$ to the current posterior $$p_{\theta_{\mathrm{old}}}(z\mid x)$$. The KL gap becomes zero, so the ELBO touches the current log likelihood.
- **M-step:** hold $$q$$ fixed and improve $$\theta$$ by maximizing the ELBO. Since $$q$$ is fixed, the term $$-\mathbb E_q[\log q(z)]$$ does not depend on $$\theta$$, leaving the expected log joint probability from the previous equation.

Imagine a lower bound touching a curve at the current parameters. The M-step raises that bound; the true log likelihood must be at least as high as the new bound. The next E-step makes the bound touch the curve again. This explains why exact EM does not decrease the observed-data likelihood. If we only approximate the E-step or fail to improve the bound, that guarantee may not hold.

The connection to Reward Forcing is the **alternating viewpoint**: construct a distribution that assigns weights to possibilities, then update the model using those weights. The next section explains how reward can be used to form such a distribution.

## Preliminary: RL from an EM Perspective

In reinforcement learning (RL), we want the model to produce samples with high reward. For video generation, a sample is a video, and its reward might measure motion quality. We also want to retain useful behavior from the original model.

Let $$x$$ be a video, $$c$$ its prompt, $$q(x\mid c)$$ the reference model's distribution, $$p(x\mid c)$$ the new distribution we want to learn, and $$r(x,c)$$ the reward. For now, consider one fixed prompt.

### From Optimality to an ELBO

We can express the goal of generating high-reward videos as a probabilistic inference problem. Introduce a variable $$O$$: $$O=1$$ means that a generated video is considered desirable. We define its likelihood so that a higher reward makes optimality more likely:

$$
P(O=1\mid x,c)\propto\exp\!\left(\frac{r(x,c)}{\beta}\right),
\qquad \beta>0.
$$

This is a modeling choice: we use the reward to define what “desirable” means. For a literal probability, assume the reward is bounded above and choose the proportionality constant so the likelihood is at most one. That constant will not affect the optimization over $$p$$.

Under the reference model, the probability of obtaining a desirable video is

$$
P(O=1\mid c)=\int q(x\mid c)P(O=1\mid x,c)\,dx.
$$

Introduce $$p(x\mid c)$$ by multiplying and dividing inside the integral by $$p(x\mid c)$$. Taking the logarithm and applying Jensen's inequality gives

$$
\begin{aligned}
\log P(O=1\mid c)
&=\log\mathbb E_{x\sim p(\cdot\mid c)}\!\left[
\frac{q(x\mid c)P(O=1\mid x,c)}{p(x\mid c)}
\right]\\
&\geq\mathbb E_{x\sim p(\cdot\mid c)}\!\left[
\log\frac{q(x\mid c)P(O=1\mid x,c)}{p(x\mid c)}
\right].
\end{aligned}
$$

The last line is an **ELBO** on the log probability of optimality. As in the earlier EM example, $$p$$ plays the role of an approximate posterior: it describes which videos we think are likely to be desirable. The equality assumes $$p$$ has support wherever the integrand contributes.

Since $$\log P(O=1\mid x,c)=r(x,c)/\beta+\text{constant}$$, we can rewrite the ELBO, dropping the constant, as

$$
\mathcal J_{\mathrm{RL}}(p,q)
=\mathbb E_{x\sim p(\cdot\mid c)}\left[
\frac{r(x,c)}{\beta}
-\log\frac{p(x\mid c)}{q(x\mid c)}
\right].
$$

The expectation of the log ratio is a KL divergence. Multiplying by the positive constant $$\beta$$ gives the equivalent optimization problem

$$
\max_p\quad
\mathbb E_{x\sim p(\cdot\mid c)}[r(x,c)]
-\beta\,D_{\mathrm{KL}}\!\left(p(\cdot\mid c)\|q(\cdot\mid c)\right).
$$

The first term encourages higher-reward videos. The second discourages $$p$$ from moving too far from the reference $$q$$. A smaller $$\beta$$ makes reward more influential relative to this constraint. For multiple prompts, we also average the objective over $$c$$. [Reward Forcing, Section 3.1](https://arxiv.org/html/2512.04678v2#S3.SS1)

### The Reward-Weighted Distribution

The ELBO is tight when $$p$$ equals the posterior distribution of videos conditioned on optimality. Using Bayes' rule gives

$$
p^*(x\mid c)=P(x\mid O=1,c)
=\frac{q(x\mid c)\exp(r(x,c)/\beta)}{Z(c)},
\qquad
Z(c)=\mathbb E_{x\sim q(\cdot\mid c)}
\left[\exp(r(x,c)/\beta)\right].
$$

In words, start with the reference distribution and give higher-reward videos more weight. $$Z(c)$$ normalizes the result into a probability distribution.

We can understand this using an **EM-like** two-step view:

1. **E-step:** form a target distribution that favors high-reward videos, such as $$p^*$$ above.
2. **M-step:** update the generator to better match that target distribution.

For intuition, suppose two videos are equally likely under the reference distribution, but one has a higher reward. The target distribution gives that video more weight, so fitting the target encourages the generator to produce it more often. The reward is used to determine the weight; this view does not require taking a gradient through the reward model.

### How Does the Paper Obtain Its M-Step Objective?

Let $$p'_{\mathrm{fake}}$$ be the **old student distribution**, fixed during the M-step. The paper uses it as the reference distribution in the E-step and calls the resulting reward-weighted distribution $$p'_{\mathrm{real}}$$:

$$
p'_{\mathrm{real}}(x\mid c)
=p'_{\mathrm{fake}}(x\mid c)
\underbrace{\frac{\exp(r(x,c)/\beta)}{Z(c)}}_{w(x,c)}.
$$

Here $$Z(c)=\mathbb E_{x\sim p'_{\mathrm{fake}}(\cdot\mid c)}[\exp(r(x,c)/\beta)]$$ normalizes the distribution. Despite its name, $$p'_{\mathrm{real}}$$ is the **reward-weighted old student distribution**; it is different from the original teacher distribution $$p_{\mathrm{real}}$$.

The names can be confusing, so it helps to compare them directly:

- $$p_{\mathrm{real}}$$ is the target distribution represented by the pretrained teacher. Its score guides ordinary DMD toward the teacher's video distribution.
- $$p'_{\mathrm{real}}$$ is an ideal reward-weighted distribution constructed from the **old student** $$p'_{\mathrm{fake}}$$. It assigns relatively more probability to student videos with higher rewards.

For example, suppose the old student generates a static video and a moving video with equal probability. If the moving video receives a higher reward, $$p'_{\mathrm{real}}$$ gives it a higher relative probability. Meanwhile, $$p_{\mathrm{real}}$$ remains the teacher's distribution. In the practical update, the teacher supplies the real score and the reward weights the DMD update; a separate model for $$p'_{\mathrm{real}}$$ is not constructed.

The M-step fits the new student $$p_\theta=p_{\mathrm{fake}}$$ to this target by maximizing its expected log likelihood:

$$
\max_\theta\;
\mathbb E_{c\sim p(c),\,x\sim p'_{\mathrm{real}}(\cdot\mid c)}
\left[\log p_\theta(x\mid c)\right].
$$

Substituting $$p'_{\mathrm{real}}=p'_{\mathrm{fake}}w$$ gives

$$
\max_\theta\;
\mathbb E_{c\sim p(c),\,x\sim p'_{\mathrm{fake}}(\cdot\mid c)}
\left[w(x,c)\log p_\theta(x\mid c)\right].
$$

In words, we can sample videos from the old student and give high-reward videos more weight when fitting the new student. The old distribution and the weights are held fixed during this ideal M-step. This maximum-likelihood objective is equivalent to minimizing $$D_{\mathrm{KL}}(p'_{\mathrm{real}}\|p_\theta)$$, averaged over prompts.

**Where does the log ratio in the paper come from?** The teacher distribution $$p_{\mathrm{real}}$$ is fixed. Therefore, when the E-step distribution is also fixed, the expectation of $$\log p_{\mathrm{real}}(x\mid c)$$ does not depend on $$\theta$$. Subtracting it changes the objective's value but not the parameters that maximize it:

$$
\begin{aligned}
\max_\theta\quad
&\mathbb E_{c,\,x\sim p'_{\mathrm{fake}}}
\left[w(x,c)\log p_\theta(x\mid c)\right] \\
\equiv\;\max_\theta\quad
&\mathbb E_{c,\,x\sim p'_{\mathrm{fake}}}
\left[w(x,c)\log\frac{p_\theta(x\mid c)}{p_{\mathrm{real}}(x\mid c)}\right].
\end{aligned}
$$

The second line is the form of the paper's displayed M-step objective. The equivalence here refers to **maximizing** either expression with the E-step distribution and teacher fixed. [Reward Forcing, Section 3.3](https://arxiv.org/html/2512.04678v2#S3.SS3)

### How Does This Connect to DMD?

The M-step's log ratio resembles the one in ordinary DMD, but the objectives have different directions and sampling distributions:

- The ideal EM M-step **maximizes** its expectation over the fixed, reward-weighted old student.
- Ordinary DMD **minimizes** a KL objective whose expectation is over the current student distribution, which changes with $$\theta$$.

Consequently, the paper's practical reward-weighted DMD update does not follow simply by differentiating the fixed-distribution M-step above. Instead, the EM perspective motivates prioritizing high-reward samples, and the paper combines that idea with an **approximate score-based DMD update**. It does not explicitly construct $$p'_{\mathrm{real}}$$ or perform an exact maximum-likelihood M-step, so the usual exact EM improvement guarantee does not automatically apply.

## Rewarded DMD

Ordinary DMD trains a generator using the difference between two estimated scores:

- The **real score**, estimated by a frozen pretrained teacher, describes the target distribution.
- The **fake score**, learned by a separate model, describes the student's current output distribution.

Let $$x_0=G_\theta(z,c)$$ be a video generated from noise $$z$$ and prompt $$c$$. We add noise to obtain $$x_t$$, then ask both score models to evaluate that same noisy video. The difference between their scores tells the student how its distribution should move toward the teacher's distribution.

Ignoring timestep-dependent factors, the ordinary DMD **gradient** has the following form:

$$
\widehat{\nabla_\theta\mathcal L_{\mathrm{DMD}}}
\;\approx\;
\mathbb E\left[
J_G^\top\!
\left(s_{\mathrm{fake}}(x_t,t,c)-s_{\mathrm{real}}(x_t,t,c)\right)
\right],
\qquad
J_G=\frac{\partial G_\theta(z,c)}{\partial\theta}.
$$

The score difference gives an update direction for the generated video. $$J_G$$ passes that direction back to the generator parameters. In practice, both scores and the resulting gradient are estimated.

Now suppose two student videos produce different motion-quality rewards. Ordinary DMD uses the score difference for each video without a motion-reward weight. Re-DMD keeps that same basic update, but multiplies each video's contribution by $$\exp(r(x_0,c)/\beta)$$. A higher-reward video therefore has more influence on the update. The paper computes this reward from decoded frames using a video reward model based on VideoAlign.

Adding this weight gives the following schematic **gradient**, rather than a directly evaluated loss:

$$
\widehat{\nabla_\theta\mathcal L_{\mathrm{Re\text{-}DMD}}}
\;\approx\;
\mathbb E\left[
\underbrace{\exp\!\left(\frac{r(x_0,c)}{\beta}\right)}_{\text{reward weight}}
\;J_G^\top\!
\left(s_{\mathrm{fake}}(x_t,t,c)-s_{\mathrm{real}}(x_t,t,c)\right)
\right],
$$

The expectation averages over generated videos, noise, and timesteps. As above, timestep-dependent factors and practical normalization are omitted for readability.

A video with a higher reward receives a larger weight on its score-difference update. The reward is treated as a fixed weight for this update, so the method does not need to backpropagate through the reward model. It is therefore more precise to view Re-DMD as a **reward-weighted approximate DMD gradient** than to assume that the code evaluates an exact KL or differentiates a reward objective end to end. [Reward Forcing, Section 3.3](https://arxiv.org/html/2512.04678v2#S3.SS3)

A useful distinction from ordinary RL fine-tuning is that the paper uses the reward to **weight the distribution-matching signal**. The model still learns from the teacher and fake-score estimates; the reward does not replace them.

## What Does the Reward Encourage?

In the paper's experiments, the reward is intended to favor videos with better motion dynamics. The authors report that removing Re-DMD reduces their measured dynamic score, while removing EMA-Sink also harms the result. This suggests that the two components contribute in different ways.

A higher motion reward is not automatically better in every respect. If reward weighting is too strong, the model may favor excessive or undesirable motion at the expense of background consistency, smoothness, or image quality. The paper reports this trade-off when varying $$\beta$$. Reward quality also depends on what the reward model can measure, so a high score is not a guarantee of an ideal video. [Reward Forcing, Section 4.2](https://arxiv.org/html/2512.04678v2#S4.SS2)


## Python-Style Pseudocode

The following is a simplified view of the authors' [Re-DMD implementation](https://github.com/JaydenLyh/Reward-Forcing/blob/main/model/re_dmd.py). It focuses on one video at a time, as in the provided training configuration. The rollout is shown as one operation because it includes several denoising steps and causal KV-cache management; the loss and model updates are shown explicitly.

```python
# Already initialized:
# student: causal generator, initialized from an ODE-distilled checkpoint
# teacher: frozen bidirectional diffusion model
# fake_model: trainable diffusion model that follows the student distribution
# vae: converts video latents into video frames
# reward_model: scores motion quality of decoded video frames
# scheduler: adds noise at a sampled diffusion timestep
# student_optimizer, fake_optimizer: separate optimizers

reward_scale = 2.0  # `beta` in the released configuration

for step, prompt in enumerate(prompts):
    # This rollout generates latent video chunks autoregressively.
    # It uses multiple denoising steps and backward simulation during training.
    # The returned latent has a gradient path to the selected student outputs.
    if step % 5 == 0:
        video_latent = student.rollout(prompt, with_grad=True)

        # The reward is computed from decoded frames, not latent frames.
        with torch.no_grad():
            frames = vae.decode(video_latent)
            motion_reward = reward_model(frames, prompt)["MQ"]
            weight = torch.exp(reward_scale * motion_reward)

            t = sample_diffusion_timestep()
            noise = torch.randn_like(video_latent)
            noisy_latent = scheduler.add_noise(video_latent, noise, t)

            # Both models predict a clean latent from the same noisy input.
            fake_clean = fake_model.predict_clean(noisy_latent, t, prompt)
            real_clean = teacher.predict_clean_with_guidance(noisy_latent, t, prompt)

            # The released code normalizes the clean-prediction difference.
            scale = (video_latent - real_clean).abs().mean()
            dmd_direction = (fake_clean - real_clean) / scale
            dmd_direction = torch.nan_to_num(dmd_direction)

        # The target is fixed: differentiating this MSE gives the desired
        # reward-weighted DMD direction for the student output.
        target = (video_latent - dmd_direction).detach()
        student_loss = 0.5 * weight * ((video_latent - target) ** 2).mean()

        student_optimizer.zero_grad()
        student_loss.backward()
        student_optimizer.step()

    # Train the fake model every iteration on a newly generated student video.
    # The student is fixed during this update.
    with torch.no_grad():
        fake_video = student.rollout(prompt, with_grad=False)
        t = sample_diffusion_timestep()
        noise = torch.randn_like(fake_video)
        noisy_video = scheduler.add_noise(fake_video, noise, t)

    # For the flow-matching schedule, the training target is noise - clean.
    predicted_flow = fake_model.predict_flow(noisy_video, t, prompt)
    target_flow = noise - fake_video
    fake_loss = ((predicted_flow - target_flow) ** 2).mean()

    fake_optimizer.zero_grad()
    fake_loss.backward()
    fake_optimizer.step()
```

`student.rollout` represents the causal, multi-step generation pipeline; `predict_clean` and `predict_flow` represent different outputs or conversions of the diffusion model. `sample_diffusion_timestep()` samples a training noise level. These names explain the operations and are not literal repository APIs. The real code also handles gradient masks, timestep shifts, classifier-free guidance, distributed training, and gradient accumulation. [Training pipeline](https://github.com/JaydenLyh/Reward-Forcing/blob/main/pipeline/reward_forcing_training.py) · [Configuration](https://github.com/JaydenLyh/Reward-Forcing/blob/main/configs/reward_forcing.yaml)

The MSE target is detached, so its gradient with respect to the student output is proportional to `weight * dmd_direction`. It is a surrogate for applying that update; its numerical value is not the KL divergence. The actual implementation applies the loss only to selected positions when a gradient mask is present.

**A notation detail:** the paper writes the reward weight as $$\exp(r/\beta)$$, whereas the released code uses `exp(beta * reward)` with `beta: 2.0`. The code's `beta` is therefore a reward multiplier, corresponding to $$1/\beta$$ in the paper's formula. The code uses the normalized motion-quality reward and does not explicitly compute $$Z(c)$$.

### Where EMA-Sink Fits

During the rollout and later during inference, the attention cache maintains recent keys and values alongside a fixed-size sink. When old entries leave the local window, the sink update has the following form:

```python
# Conceptual cache update, performed when old KV entries leave the window.
sink_keys = alpha * sink_keys + (1 - alpha) * evicted_keys
sink_values = alpha * sink_values + (1 - alpha) * evicted_values

# New queries attend to the updated sink and the recent local cache.
attention_keys = concatenate(sink_keys, recent_keys)
attention_values = concatenate(sink_values, recent_values)
```

This illustrates the EMA-Sink equation rather than a literal copy of the cache implementation. The sink stores compressed **keys and values**, while the reward model evaluates decoded **video frames**; they serve different purposes.
