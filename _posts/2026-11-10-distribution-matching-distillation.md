---
layout: post
categories: [diffusion-models]
title: "3. Distribution Matching Distillation"
date: 2026-11-10 12:00:00 +0100
description: "From VSD to one-step generation: the two scores, regression loss, and practical surrogate gradients behind DMD."
math: true
---

[One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)

This is a paper I appreciate a lot. The idea is elegant: instead of forcing a student to reproduce every step of a diffusion model's sampling trajectory, train it to produce samples from the same distribution in one step. I find the connection to **Variational Score Distillation (VSD)** especially helpful. When I first read DMD, it was not immediately clear why another diffusion model was needed to learn the student's current distribution. Understanding VSD made the role of that second model much more natural.

DMD has also influenced later work on fast diffusion generation. I find the broader idea particularly interesting for autoregressive video generation, where related distillation methods can transfer knowledge from a bidirectional teacher to a faster causal student. These later methods introduce their own training and conditioning choices; the original DMD paper focuses on one-step image generation.

<nav class="dmd-guide" aria-label="In this article">
  <p class="dmd-guide-label">In this article</p>
  <ul>
    <li><a href="#the-basic-idea">The basic idea</a></li>
    <li><a href="#distribution-matching-loss">Distribution matching</a></li>
    <li><a href="#regression-loss">Regression loss</a></li>
    <li><a href="#dmd-pseudocode">Training pseudocode</a></li>
    <li><a href="#an-mse-surrogate">The MSE surrogate</a></li>
  </ul>
</nav>

## The Basic Idea

A diffusion model usually transforms noise into an image through many denoising steps. DMD distills this process into a generator $$G_\theta$$ that maps Gaussian noise directly to a sample:

$$
z\sim\mathcal N(0,I),\qquad x_0=G_\theta(z).
$$

For a conditional model, such as text-to-image generation, the generator and score models also receive a condition $$c$$. I omit $$c$$ from most equations for readability; the distributions should then be understood as conditional on it.

By analogy with GANs, the paper calls the generator's output distribution $$p_{\mathrm{fake}}$$ and the target data distribution $$p_{\mathrm{real}}$$. The generator is initialized from the pretrained diffusion model. A fixed copy of that model estimates the real score, while another copy is fine-tuned to estimate the fake score as the generator changes. The fake-score model is an auxiliary denoiser, not the one-step generator itself.

The distribution-matching part of DMD does not require a particular noise sample $$z$$ to produce the same image as the teacher. The full original method also includes a paired regression loss, however, which does encourage that correspondence. This combination is important: distribution matching improves realism, while regression helps stabilize training and preserve diversity. [DMD, Section 3](https://arxiv.org/html/2311.18828v4#S3)

## Distribution Matching Loss

Ideally, we would minimize the reverse KL divergence

$$
D_{\mathrm{KL}}(p_{\mathrm{fake}}^\theta\|p_{\mathrm{real}})
=\mathbb E_{x\sim p_{\mathrm{fake}}^\theta}
\left[\log p_{\mathrm{fake}}^\theta(x)-\log p_{\mathrm{real}}(x)\right].
$$

The order matters: the expectation is over student samples. Although these densities are generally intractable, we can express the gradient using their **scores**, the gradients of their log densities with respect to the sample. Under suitable regularity assumptions, if $$J_G=\partial G_\theta(z)/\partial\theta$$,

$$
\nabla_\theta D_{\mathrm{KL}}
=\mathbb E_z\left[J_G^\top
\left(s_{\mathrm{fake}}(x)-s_{\mathrm{real}}(x)\right)\right],
\qquad x=G_\theta(z),
$$

where $$s_{\mathrm{fake}}=\nabla_x\log p_{\mathrm{fake}}^\theta$$ and $$s_{\mathrm{real}}=\nabla_x\log p_{\mathrm{real}}$$. The explicit derivative of $$\log p_{\mathrm{fake}}^\theta$$ with respect to $$\theta$$ has zero expectation: it is the expectation of the density's parameter score. This leaves the pathwise derivative above.

During gradient descent, the sign reverses: the real score pulls generated samples toward regions favored by the target, while the fake score accounts for the distribution already produced by the generator. The latter can encourage the samples to spread out, rather than all following the real score toward the same nearby mode. This is an intuition for the update, not a guarantee against mode collapse.

### Why Add Noise?

Clean image distributions may occupy a small, structured part of image space. A generated sample can lie far from the real data manifold, where the clean real score is unreliable or even undefined. Moreover, a diffusion model naturally estimates scores at noisy timesteps. DMD therefore compares the distributions after applying the same Gaussian corruption:

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

For $$\sigma_t>0$$, this smoothing gives the noisy distributions overlapping support. Let $$p_{\mathrm{fake},t}^\theta$$ and $$p_{\mathrm{real},t}$$ be the resulting noisy distributions. For a fixed timestep, the gradient of their KL divergence is

$$
\nabla_\theta D_{\mathrm{KL}}
\left(p_{\mathrm{fake},t}^\theta\|p_{\mathrm{real},t}\right)
=\mathbb E_{z,\epsilon}\left[
\alpha_t J_G^\top
\left(s_{\mathrm{fake}}(x_t,t)-s_{\mathrm{real}}(x_t,t)\right)
\right].
$$

The factor $$\alpha_t$$ comes from differentiating $$x_t$$ through $$x_0=G_\theta(z)$$. DMD samples different noise levels and replaces the exact scores with diffusion-model estimates. This produces a practical update inspired by noisy distribution matching; it is not the exact gradient of the clean-image KL above. [DMD, Section 3.2 and Appendix F](https://arxiv.org/html/2311.18828v4#S3.SS2)

### Estimating the Two Scores

For a noise-prediction model, the relation between a denoiser and the noisy score is

$$
s(x_t,t)\approx-\frac{\epsilon_{\mathrm{pred}}(x_t,t)}{\sigma_t}.
$$

The frozen pretrained model supplies $$\epsilon_{\mathrm{real}}$$. To estimate $$\epsilon_\phi$$, the fake-score model is trained on current generator samples with the ordinary diffusion denoising loss:

$$
\mathcal L_{\mathrm{fake}}(\phi)
=\mathbb E_{z,t,\epsilon}\left[
\lambda(t)\left\|
\epsilon_\phi(\alpha_t\operatorname{stopgrad}(G_\theta(z))
+\sigma_t\epsilon,t)-\epsilon
\right\|_2^2\right].
$$

The generator output is detached in this update, so this loss trains the fake denoiser rather than the generator. The paper expresses the same idea using clean-image prediction; the appropriate loss weighting depends on the parameterization. The fake denoiser must keep tracking the generator because $$p_{\mathrm{fake}}^\theta$$ changes throughout training.

Substituting noise predictions into the score difference gives the generator's distribution-matching gradient estimate:

$$
\widehat{\nabla_\theta\mathcal L_{\mathrm{DM}}}
=\mathbb E_{z,t,\epsilon}\left[
J_G^\top\,w_t\frac{\alpha_t}{\sigma_t}
\left(\epsilon_{\mathrm{real}}(x_t,t)
-\epsilon_\phi(x_t,t)\right)
\right].
$$

This resembles VSD: both updates use a pretrained target score and a learned score for the current distribution. VSD originally evolves a distribution of 3D scenes through rendered images; DMD uses the score difference to train an image generator.

DMD introduces a weight to normalize update magnitudes across noise levels. In the paper's clean-image-prediction notation, it is proportional to $$\sigma_t^2/\alpha_t$$ and inversely proportional to the mean absolute difference between the teacher's denoised prediction and the generated sample. Because this normalization also depends on the sample, the resulting update should be understood as a weighted gradient estimator, rather than assumed to be the exact gradient of a simple weighted KL objective. The score predictions and weight are treated as fixed when applying the update to the generator. [DMD, Equations 7–8](https://arxiv.org/html/2311.18828v4#S3.SS2)

## Regression Loss

Distribution matching alone can still be unstable and miss modes in practice. DMD adds a regression loss using a precomputed dataset of pairs $$(z,y)$$: $$z$$ is initial Gaussian noise and $$y$$ is the output of the pretrained teacher's deterministic multi-step sampler starting from that same noise. The loss is

$$
\mathcal L_{\mathrm{reg}}(\theta)
=\mathbb E_{(z,y)\sim\mathcal D}
\left[\operatorname{LPIPS}(G_\theta(z),y)\right].
$$

LPIPS is a perceptual distance. The paired loss encourages the student to preserve the large-scale structure of the teacher's output for a given seed, while the distribution-matching update improves the realism of its samples. It helps reduce mode dropping in the paper's experiments, but it does not mathematically guarantee that every mode is preserved.

The generator is trained with the distribution-matching update plus $$\lambda_{\mathrm{reg}}\nabla_\theta\mathcal L_{\mathrm{reg}}$$. The teacher trajectories are computed in advance, so the full sampling process does not need to run for every training update. Constructing the paired dataset still has a cost, and the regression term ties the student to the teacher's particular sampling paths. [DMD, Section 3.3](https://arxiv.org/html/2311.18828v4#S3.SS3)


## DMD Pseudocode

The following Python-style pseudocode summarizes the original DMD training loop. It uses noise prediction to match the equations above; the paper writes the algorithm in terms of clean-image prediction. Helpers for sampling, conditioning, the noise schedule, and LPIPS are left abstract. For a conditional model, use the same condition for a generated sample and both score predictions. [DMD, Algorithm 1](https://arxiv.org/html/2311.18828v4#S3.SS3)

```python
import torch


def train_dmd(teacher, paired_data, steps, lambda_reg):
    # Separate copies: the teacher stays fixed throughout training.
    generator = initialize_generator_from(teacher)
    fake_denoiser = copy_denoiser(teacher)
    teacher.requires_grad_(False)
    teacher.eval()

    generator_optimizer = make_optimizer(generator.parameters())
    fake_optimizer = make_optimizer(fake_denoiser.parameters())

    for _ in range(steps):
        # The pairs were generated beforehand by a deterministic teacher sampler.
        z_ref, y_ref, c_ref = paired_data.sample()
        z, c = sample_noise_and_condition()
        x0 = generator(z, c)
        x_ref = generator(z_ref, c_ref)

        # Distribution-matching update for the generator.
        t = sample_dm_timestep()  # Avoid the endpoints of the noise schedule.
        noise = torch.randn_like(x0)
        with torch.no_grad():
            x_t = alpha(t) * x0 + sigma(t) * noise
            eps_real = teacher(x_t, t, c)  # May include fixed-scale guidance.
            eps_fake = fake_denoiser(x_t, t, c)

            # Convert the teacher's noise prediction to a clean prediction.
            x_real = (x_t - sigma(t) * eps_real) / alpha(t)
            mean_error = mean_over_nonbatch_dims(abs(x_real - x0))
            weight = (sigma(t) ** 2 / alpha(t)) / mean_error.clamp_min(1e-6)
            grad_x = weight * alpha(t) / sigma(t) * (eps_real - eps_fake)

        # This surrogate supplies grad_x as the gradient with respect to x0.
        loss_dm = (x0 * grad_x).flatten(1).sum(dim=1).mean()
        loss_reg = lpips(x_ref, y_ref).mean()
        generator_optimizer.zero_grad()
        (loss_dm + lambda_reg * loss_reg).backward()
        generator_optimizer.step()

        # Train the fake denoiser on detached student samples.
        s = sample_denoising_timestep()
        fake_noise = torch.randn_like(x0)
        x_s = alpha(s) * x0.detach() + sigma(s) * fake_noise
        eps_pred = fake_denoiser(x_s, s, c)
        per_sample_error = mean_over_nonbatch_dims((eps_pred - fake_noise) ** 2)
        loss_fake = (denoising_weight(s) * per_sample_error).mean()
        fake_optimizer.zero_grad()
        loss_fake.backward()
        fake_optimizer.step()

    return generator
```

The schedule coefficients and per-sample weights are assumed to broadcast over image dimensions. The small clamp keeps the normalization finite when the teacher's prediction is very close to $$x_0$$. The loss `loss_dm` is a gradient surrogate: its value is not the KL divergence. Because `grad_x` is computed without gradient tracking, differentiating the surrogate produces the desired score-difference update through the generator. The fake-denoiser loss uses detached samples, so it updates only the auxiliary model. In a latent diffusion implementation, the generator and denoisers operate in latent space and the regression distance must be evaluated in the appropriate image space.

### Why Does the Surrogate Loss Give the KL Gradient?

The distribution-matching loss in the pseudocode is designed to produce a particular gradient when we call `.backward()`:

```python
with torch.no_grad():
    grad_x = weight * alpha(t) / sigma(t) * (eps_real - eps_fake)

loss_dm = (x0 * grad_x).flatten(1).sum(dim=1).mean()
loss_dm.backward()
```

Its numerical value is not a KL divergence. To see what it does, write the loss for one sample as

$$
\mathcal L_{\mathrm{surrogate}}=x_0^\top d,
\qquad d=\operatorname{stopgrad}(\texttt{grad\_x}),
\qquad x_0=G_\theta(z).
$$

Because $$d$$ is treated as a constant during backpropagation, the chain rule gives

$$
\nabla_\theta\mathcal L_{\mathrm{surrogate}}
=\left(\frac{\partial G_\theta(z)}{\partial\theta}\right)^\top d
=J_G^\top d.
$$

In other words, `grad_x` specifies the gradient with respect to the generated sample $$x_0$$, and `.backward()` propagates it through the generator. The sum in `loss_dm` takes the inner product over the sample's dimensions; the mean averages over the batch.

The connection to KL comes from the score identity derived earlier. At a fixed timestep $$t$$, assuming exact scores and suitable regularity conditions,

$$
\nabla_\theta D_{\mathrm{KL}}
\left(p_{\mathrm{fake},t}^\theta\|p_{\mathrm{real},t}\right)
=\mathbb E_{z,\epsilon}\left[
J_G^\top\alpha_t
\left(s_{\mathrm{fake}}(x_t,t)-s_{\mathrm{real}}(x_t,t)\right)
\right].
$$

Using $$s(x_t,t)\approx-\epsilon_{\mathrm{pred}}(x_t,t)/\sigma_t$$,

$$
s_{\mathrm{fake}}-s_{\mathrm{real}}
\approx\frac{\epsilon_{\mathrm{real}}-\epsilon_{\mathrm{fake}}}{\sigma_t}.
$$

This is why the code sets

$$
d=w_t\frac{\alpha_t}{\sigma_t}
\left(\epsilon_{\mathrm{real}}-\epsilon_{\mathrm{fake}}\right).
$$

Without the extra weight, and with exact scores, the expected surrogate gradient equals the gradient of the noisy KL at that timestep. A weight depending only on $$t$$ yields the gradient of the corresponding weighted average of noisy KL objectives. The practical DMD update is approximate: the two models estimate the scores, and the paper's normalization weight also depends on the sample. Since that weight is held fixed during the update, we should not generally interpret the resulting update as the exact gradient of a simple weighted KL. It is a weighted gradient estimator inspired by distribution matching.

Stopping the gradient through `grad_x` is essential: otherwise automatic differentiation would introduce derivatives through the score predictions and the weight, adding terms that are not part of the intended update. The regression loss is separate and contributes its own ordinary gradient to the generator update.

### An MSE Surrogate

We can also produce the same generator gradient with a squared-error surrogate. Construct a target by taking one step from the current sample in the negative `grad_x` direction, and detach the **entire target**:

```python
target = (x0 - grad_x).detach()
loss_dm = 0.5 * (x0 - target).square().flatten(1).sum(dim=1).mean()
```

For one sample, let $$d=\texttt{grad\_x}$$. The loss is

$$
x_{\mathrm{target}}=\operatorname{stopgrad}(x_0-d),
\qquad
\mathcal L_{\mathrm{MSE}}=\frac12\|x_0-x_{\mathrm{target}}\|_2^2.
$$

Since the target is detached, its derivative is zero, and at the current generator parameters,

$$
\nabla_{x_0}\mathcal L_{\mathrm{MSE}}
=x_0-x_{\mathrm{target}}=d,
\qquad
\nabla_\theta\mathcal L_{\mathrm{MSE}}=J_G^\top d.
$$

Thus, this surrogate gives the same generator gradient as the inner-product loss in the pseudocode. Recompute the target for each update; its numerical loss value is not the KL divergence. The code sums over each sample's dimensions and averages over the batch, matching the earlier gradient scaling. By contrast, PyTorch's default `mse_loss(..., reduction="mean")` averages over all elements, including image dimensions, so with the same factor of $$1/2$$ its gradient is smaller by the number of elements in each sample.
