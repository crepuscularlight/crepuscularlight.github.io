---
layout: post
categories: [diffusion-models]
title: "1. Score Distillation Sampling"
date: 2026-11-08 12:00:00 +0100
description: How a frozen diffusion model guides a 3D representation, from the SDS gradient and KL interpretation to an autodiff implementation.
math: true
---

[DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)

DreamFusion is a paper I find especially impressive. Its authors include Ben Poole and Ben Mildenhall, one of the authors of NeRF. What interests me most is how a pretrained diffusion model can provide a training signal for another model. I want to write down my understanding of this idea before exploring VSD, DMD, and later work on video generation and forcing methods. These papers address different problems, but understanding this shared idea seems like a useful starting point.

In September 2022, when DreamFusion appeared, text-to-image generation was advancing rapidly, while large-scale labeled 3D data remained comparatively scarce. DreamFusion used a pretrained Imagen model to guide a NeRF through its rendered images, without requiring 3D training data. SDS, or **Score Distillation Sampling**, is the mechanism that makes this possible. [DreamFusion](https://arxiv.org/abs/2209.14988)

I still wonder how much the cost and quality of 3D data constrain progress. For this post, though, I want to focus on the question that makes this paper interesting to me: **how can a model trained to denoise images tell us how to update a 3D representation?**

## From a rendered image to a denoising objective

Let the image be

$$
x = g(\theta),
$$

where $$g$$ is a differentiable renderer and $$\theta$$ contains the parameters we want to optimize. In practice, rendering also depends on a camera $$c$$, so we would write $$x=g(\theta,c)$$. I will hold the camera fixed in the derivation and return to it later.

Let $$y$$ denote the text prompt. We add Gaussian noise at timestep $$t$$:

$$
z_t = \alpha_t x + \sigma_t\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I).
$$

Here, $$\alpha_t$$ scales the image and $$\sigma_t$$ controls the noise level. A frozen diffusion model with parameters $$\phi$$ predicts the noise, $$\epsilon_\phi(z_t,t,y)$$. The corresponding clean-image estimate is

$$
\hat{x}_\phi
=
\frac{z_t-\sigma_t\epsilon_\phi(z_t,t,y)}{\alpha_t}.
$$

Substituting the expression for $$z_t$$ gives

$$
\hat{x}_\phi-x
=
\frac{\sigma_t}{\alpha_t}
\left(\epsilon-\epsilon_\phi(z_t,t,y)\right),
$$

and therefore, for a fixed timestep with $$\alpha_t,\sigma_t>0$$,

$$
\|\hat{x}_\phi-x\|_2^2
=
\frac{\sigma_t^2}{\alpha_t^2}
\|\epsilon_\phi(z_t,t,y)-\epsilon\|_2^2.
$$

This makes my original intuition more precise: a small noise-prediction error means that the denoised estimate is close to the current image at that noise level. It does **not** prove that the image is realistic or close to an independently generated diffusion sample.

A natural starting point is to optimize the rendered image using a denoising objective:

$$
\mathcal{L}_{\mathrm{denoise}}(\theta)
=
\mathbb{E}_{t,\epsilon}
\left[
\frac{\lambda(t)}{2}
\|\epsilon_\phi(z_t,t,y)-\epsilon\|_2^2
\right].
$$

The diffusion model is frozen; only $$\theta$$ changes.

## Why differentiating this loss is awkward

Define the Jacobians

$$
J_g=\frac{\partial x}{\partial\theta},
\qquad
J_\epsilon=\frac{\partial\epsilon_\phi(z_t,t,y)}{\partial z_t}.
$$

Applying the chain rule gives

$$
\nabla_\theta\mathcal{L}_{\mathrm{denoise}}
=
\mathbb{E}_{t,\epsilon}
\left[
\lambda(t)\alpha_t
J_g^\top J_\epsilon^\top
\left(\epsilon_\phi(z_t,t,y)-\epsilon\right)
\right].
$$

Even when the diffusion model's parameters are frozen, computing gradients with respect to its input requires backpropagation through the denoiser. We do not need to construct the full Jacobian explicitly, but computing the required Jacobian-vector product is still costly.

There is also a curvature issue. Under the ideal score interpretation,

$$
\epsilon_\phi(z_t,t,y)
\approx
-\sigma_t\nabla_{z_t}\log p_{\phi,t}(z_t\mid y),
$$

so

$$
J_\epsilon
\approx
-\sigma_t\nabla_{z_t}^2\log p_{\phi,t}(z_t\mid y).
$$

This is the Hessian of the log density with respect to the **noisy image**, not the Hessian with respect to the NeRF parameters. Its curvature can amplify some directions of the update and suppress others, making optimization poorly conditioned.

The striking step in SDS is to omit the denoiser Jacobian and use the residual directly:

$$
\boxed{
\nabla_\theta\mathcal{L}_{\mathrm{SDS}}
=
\mathbb{E}_{t,\epsilon}
\left[
w(t)J_g^\top
\left(\hat{\epsilon}_\phi(z_t,t,y)-\epsilon\right)
\right].
}
$$

The hat denotes classifier-free guidance. I use the common convention

$$
\hat{\epsilon}_\phi
=
\epsilon_\phi(z_t,t,\varnothing)
+
s\left[
\epsilon_\phi(z_t,t,y)
-
\epsilon_\phi(z_t,t,\varnothing)
\right],
$$

where $$s=1$$ gives the conditional prediction and $$s>1$$ strengthens guidance. The weight $$w(t)$$ incorporates timestep-dependent scaling. This update and its KL interpretation are given in [DreamFusion, Section 2.1 and Appendix A.4](https://arxiv.org/html/2209.14988v1#A4).

Dropping the Jacobian changes the objective: this is no longer the exact gradient of the denoising squared error above. To understand why this update makes sense, I find it helpful to interpret it through a distribution-matching objective.

## The core insight: what the KL divergence compares

> **SDS optimizes the rendered image so that its distribution after forward noising approaches the pretrained diffusion model's text-conditioned distribution at the same noise level.**

The forward noising process stays fixed. What changes is the image $$x=g(\theta)$$, through updates to the 3D parameters $$\theta$$.

At a fixed noise level $$t$$, repeatedly adding Gaussian noise to the same image produces the distribution

$$
q_t^\theta(z)
=
\mathcal{N}\!\left(z;\alpha_tg(\theta),\sigma_t^2I\right).
$$

The KL objective compares two distributions:

- **$$q_t^\theta$$:** noisy versions of our current rendered image.
- **$$p_{\phi,t}(\cdot\mid y)$$:** the pretrained diffusion model's distribution at the same noise level, conditioned on the text prompt $$y$$.

SDS encourages the first distribution to approach the second across noise levels. Writing the objective for an unguided teacher gives:

$$
\boxed{
\mathcal{L}_{\mathrm{SDS}}(\theta)
=
\mathbb{E}_t
\left[
\frac{\sigma_t}{\alpha_t}w(t)
D_{\mathrm{KL}}\!\left(
q_t^\theta
\,\|\,p_{\phi,t}(\cdot\mid y)
\right)
\right].
}
$$

**Updating $$\theta$$ moves the center of the noisy-render distribution toward regions preferred by the pretrained model.** The noise schedule and the pretrained model remain fixed. This is one KL divergence between two distributions at each noise level, averaged over $$t$$.

The link to the update above is simple: **the difference between predicted noise and added noise represents a difference between scores**. A score points in the direction of increasing log probability. This feedback lets us optimize the KL objective without evaluating the teacher's density or backpropagating through its denoiser. With classifier-free guidance, the update uses the guided score instead. [DreamFusion, Section 2.1 and Appendix A.4](https://arxiv.org/html/2209.14988v1#A4)

I think this derivation helped inspire later work such as VSD and DMD. What stands out to me is how a distribution-matching objective becomes a practical update based on the difference between two scores.

## Turning the update into a loss for autodiff

A convenient surrogate is

$$
\mathcal{L}_{\mathrm{surrogate}}
=
\mathbb{E}_{t,\epsilon}
\left[
w(t)
\left\langle
\operatorname{stopgrad}
\left[\hat{\epsilon}_\phi(z_t,t,y)-\epsilon\right],
g(\theta)
\right\rangle
\right].
$$

The stop-gradient operation holds the residual fixed during backpropagation. Differentiating this expression produces the SDS gradient through $$g$$, without differentiating through the diffusion model. This surrogate implements the desired update; its scalar value does not measure the KL divergence.

For a 3D model, we also average over camera views:

$$
\nabla_\theta\mathcal{L}_{\mathrm{SDS}}
=
\mathbb{E}_{c,t,\epsilon}
\left[
w(t)J_{g(\theta,c)}^\top
\left(
\hat{\epsilon}_\phi(
\alpha_tg(\theta,c)+\sigma_t\epsilon,t,y)
-\epsilon
\right)
\right].
$$

Every sampled view updates the same parameters. My interpretation is that the renderer connects the image-level feedback across viewpoints. Still, plausible individual images do not guarantee correct geometry or consistency between views.

## SDS in pseudocode

The loop below implements the SDS update above. Here, `y` is the text prompt, `s` is the guidance scale, and `w(t)` is the timestep weight.

```python
theta = initialize_3d_model()
freeze(diffusion_model)
optimizer = create_optimizer(theta)

for step in range(num_steps):
    c = sample_camera()
    t = sample_timestep(t_min, t_max)
    x = render(theta, c)                  # Keep gradients through the renderer
    epsilon = sample_standard_normal_like(x)

    with no_grad():                      # No gradients through the denoiser
        z_t = alpha(t) * x + sigma(t) * epsilon
        eps_cond = diffusion_model(z_t, t, y)
        eps_uncond = diffusion_model(z_t, t, empty_prompt)
        eps_guided = eps_uncond + s * (eps_cond - eps_uncond)
        feedback = w(t) * (eps_guided - epsilon)

    loss = sum(stop_gradient(feedback) * x)
    optimizer.zero_grad()
    loss.backward()                      # Gradients reach theta through the renderer
    optimizer.step()

return theta
```

**The diffusion model supplies the feedback; the renderer carries that feedback to the 3D parameters.** Each iteration samples a new view, noise level, and noise realization. This shows the core SDS loop; a full DreamFusion implementation also uses rendering choices and regularization to improve geometry.
