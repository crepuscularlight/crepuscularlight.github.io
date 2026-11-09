---
layout: post
title: "2. Variational Score Distillation: From SDS to DMD"
date: 2026-11-09 12:00:00 +0100
description: "Understanding VSD through variational inference, its connection to SDS and DMD, and a PyTorch-style implementation of the two alternating updates."
math: true
---

I started reading **Variational Score Distillation (VSD)** to understand **Distribution Matching Distillation (DMD)**. The part I wanted to make intuitive was the use of two diffusion models: one describes the target distribution, while the other learns the distribution currently produced by the student. Their difference tells us how to change the student.

I also admire Professor Jun Zhu's group at Tsinghua University for its influential research and the excellent researchers it has trained. I find VSD particularly inspiring: introducing another model to learn the student's distribution helped me understand why distribution matching needs two score estimates.

VSD was introduced in *ProlificDreamer* by Zhengyi Wang, Cheng Lu, and collaborators, including Jun Zhu, at NeurIPS 2023. Its original application is text-to-3D generation, where score distillation sampling (SDS) can produce oversaturated colors, overly smooth details, and limited diversity. The paper formulates optimization over a distribution of 3D scenes and derives an update using two scores. [ProlificDreamer](https://arxiv.org/abs/2305.16213)

The connection to distillation is explicit: the DMD paper credits VSD as an inspiration for training a generator through distribution matching. I will work through the mathematical connection below, keeping the distinction between a 3D scene, its rendered images, and a learned image generator clear. [DMD, Introduction](https://arxiv.org/html/2311.18828v4#S1)

## Sampling from 3D Distribution as Variational Inference

The first step is to treat the 3D parameters $$\theta$$ as a random variable with distribution $$\mu(\theta\mid y)$$, rather than optimizing only a single fixed $$\theta$$. Here, $$y$$ is the text prompt. Consequently, the rendered image $$x_0=g(\theta,c)$$ is also a random variable, with distribution $$q_0^\mu(x_0\mid c,y)$$, where $$g$$ is the differentiable renderer and $$c$$ is the camera pose.

As in the KL-divergence interpretation of SDS, the aim is to match the rendered-image distribution to the distribution represented by a pretrained diffusion model. The difference is that the optimization variable is now the distribution $$\mu$$. For a given camera view, the paper writes this as

$$
\min_\mu D_{\mathrm{KL}}\!\left(
q_0^\mu(x_0\mid c,y)\;\|\;p_0(x_0\mid y^c)
\right),
$$

where $$p_0$$ is the pretrained model's clean-image distribution and $$y^c$$ is the text prompt augmented with view information. This is Equation 4: a variational inference problem in which the distribution induced by $$\mu$$ approximates the target image distribution.

Directly optimizing this objective is difficult. The paper therefore introduces a collection of objectives comparing the corresponding noisy distributions. Both distributions are corrupted by the same forward diffusion process,

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),
$$

and Equation 5 combines their KL divergences across timesteps and camera views:

$$
\mu^*=\arg\min_\mu\;
\mathbb{E}_{t,c}\left[
\frac{\sigma_t}{\alpha_t}\omega(t)
D_{\mathrm{KL}}\!\left(
q_t^\mu(x_t\mid c,y)\;\|\;p_t(x_t\mid y^c)
\right)
\right].
$$

Here, $$\omega(t)$$ weights the different noise levels. The precise connection to the original objective is given by Theorem 1: at a noise level with $$\alpha_t,\sigma_t>0$$, the noisy KL is zero if and only if the clean distributions match. Thus, when exact matching is achievable, the objectives share that optimum. Their KL values are not generally equal, and the theorem does not establish that their best approximations coincide when exact matching is impossible. [ProlificDreamer, Section 3.1, Equations 4–6](https://arxiv.org/html/2305.16213v2#S3.SS1)

To solve this distribution-optimization problem, a direct approach would be to train another parameterized generative model for $$\mu$$. However, this could introduce substantial computational cost and optimization complexity. The paper instead represents $$\mu$$ using a set of 3D parameter particles, $$\{\theta^{(i)}\}_{i=1}^n$$, and derives a rule for evolving them.

Starting from an initial sample $$\theta_0\sim\mu_0(\theta_0\mid y)$$, Equation 7 describes how to move the particle through a Wasserstein gradient flow. Writing $$J_g=\partial g(\theta_\tau,c)/\partial\theta_\tau$$, the equation can be expressed as

$$
\frac{d\theta_\tau}{d\tau}
=\mathbb{E}_{t,\epsilon,c}\left[
\sigma_t\omega(t)J_g^\top
\left(
\nabla_{x_t}\log p_t(x_t\mid y^c)
-\nabla_{x_t}\log q_t^{\mu_\tau}(x_t\mid c,y)
\right)
\right],
$$

where $$x_t=\alpha_tg(\theta_\tau,c)+\sigma_t\epsilon$$. Here, $$\tau$$ measures optimization progress, while $$t$$ is the diffusion timestep. Simulating this flow moves the initial particles toward a distribution minimizing the VSD objective; the practical algorithm approximates the flow with finite optimization steps and learned scores.

From this equation, we can see why a second diffusion model is introduced. There are two score functions, where a score means the gradient of a log density with respect to $$x_t$$. The first describes the noisy target-image distribution and can be estimated by the pretrained diffusion model. The second describes the current noisy rendered-image distribution and must be learned as the particles evolve. In the noise-prediction parameterization,

$$
-\sigma_t\nabla_{x_t}\log p_t(x_t\mid y^c)
\approx\epsilon_{\mathrm{pretrain}}(x_t,t,y^c),
$$

$$
-\sigma_t\nabla_{x_t}\log q_t^{\mu_\tau}(x_t\mid c,y)
\approx\epsilon_\phi(x_t,t,c,y).
$$

The auxiliary model $$\epsilon_\phi$$ is trained on the current rendered images with a standard denoising objective. Equation 8 can be written, up to normalization over particles, as

$$
\mathcal{L}_{\mathrm{aux}}(\phi)
=\mathbb{E}_{i,c,t,\epsilon}\left[
\left\|
\epsilon_\phi\!\left(
\alpha_tg(\theta^{(i)},c)+\sigma_t\epsilon,t,c,y
\right)-\epsilon
\right\|_2^2
\right].
$$

The paper uses low-rank adaptation (LoRA) of the pretrained diffusion model to estimate this second score efficiently, adding camera information to the conditioning. This auxiliary model estimates the rendered-image score; it is not the separate generator of 3D parameters considered above.

Substituting these two score estimates into Equation 7 gives the VSD update in Equation 9:

$$
\nabla_\theta\mathcal{L}_{\mathrm{VSD}}(\theta)
\triangleq\mathbb{E}_{t,\epsilon,c}\left[
\omega(t)J_g^\top
\left(
\epsilon_{\mathrm{pretrain}}(x_t,t,y^c)
-\epsilon_\phi(x_t,t,c,y)
\right)
\right],
$$

with $$J_g=\partial g(\theta,c)/\partial\theta$$ and $$x_t=\alpha_tg(\theta,c)+\sigma_t\epsilon$$. The paper specifies $$\mathcal{L}_{\mathrm{VSD}}$$ through this gradient for updating the scene parameters. The particles are updated by $$\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}_{\mathrm{VSD}}$$, alternating with updates to $$\phi$$ so that the auxiliary model tracks the changing rendered-image distribution. The pretrained model remains fixed. [ProlificDreamer, Section 3.2, Equations 7–9](https://arxiv.org/html/2305.16213v2#S3.SS2)

This is the part I find especially useful for understanding DMD: the additional diffusion model estimates the current distribution, giving us the second score needed to derive the update. In VSD, that update evolves the 3D parameters; DMD uses the same score-difference idea to train a generator. [DMD, Section 3.2](https://arxiv.org/html/2311.18828v4#S3.SS2)

## Comparison with SDS

Comparing the SDS and VSD update equations, we can see that SDS is recovered when $$\mu$$ is a Dirac distribution and its noisy-image score is evaluated exactly. Let

$$
\mu(\theta\mid y)=\delta(\theta-\bar\theta).
$$

For a fixed camera, the rendered image is then deterministic, so its noisy distribution is

$$
q_t^\mu(x_t\mid c,y)
=\mathcal{N}\!\left(x_t;\alpha_tg(\bar\theta,c),\sigma_t^2I\right).
$$

Its score can be computed directly:

$$
-\sigma_t\nabla_{x_t}\log q_t^\mu(x_t\mid c,y)
=\frac{x_t-\alpha_tg(\bar\theta,c)}{\sigma_t}
=\epsilon.
$$

Replacing $$\epsilon_\phi$$ with this exact expression in the VSD gradient gives the SDS gradient:

$$
\nabla_\theta\mathcal{L}_{\mathrm{SDS}}(\theta)
\approx\mathbb{E}_{t,\epsilon,c}\left[
\omega(t)J_g^\top
\left(\epsilon_{\mathrm{pretrain}}(x_t,t,y^c)-\epsilon\right)
\right].
$$

This is the sense in which SDS is a special case of VSD. A practical VSD implementation can still use one particle while learning a neural score approximation; that learned prediction need not equal the analytic Dirac-case expression at every optimization step. Therefore, using one particle alone does not make the practical algorithm identical to SDS. [ProlificDreamer, Section 3.3](https://arxiv.org/html/2305.16213v2#S3.SS3)

## VSD Pseudocode

The following summarizes the alternating updates in the paper's Algorithm 1, using Python-style pseudocode with PyTorch notation. The initialization, rendering, sampling, and noise-schedule helpers are left abstract. [ProlificDreamer, Appendix E](https://arxiv.org/html/2305.16213v2)

```python
import torch
import torch.nn.functional as F
from torch.optim import SGD


def vsd(y, teacher, render, n, steps, lr_theta, lr_phi):
    teacher.requires_grad_(False)
    particles = initialize_particles(n, y)  # Sample scenes from mu_0.
    auxiliary = initialize_lora_denoiser(teacher, camera_conditioning=True)
    scene_optimizers = [
        SGD(scene.parameters(), lr=lr_theta) for scene in particles
    ]
    aux_optimizer = SGD(auxiliary.parameters(), lr=lr_phi)

    for _ in range(steps):
        i = sample_particle_index(n)  # Uniform over particles.
        c = sample_camera()
        y_c = make_view_prompt(y, c)
        x0 = render(particles[i], c)
        x_clean = x0.detach()

        # Update the selected scene particle (Equation 9).
        t = sample_distillation_timestep()
        noise = torch.randn_like(x0)
        with torch.no_grad():
            x_t = alpha(t) * x0 + sigma(t) * noise
            eps_target = teacher(x_t, t, y_c)
            eps_current = auxiliary(x_t, t, c, y)
            grad_x = omega(t) * (eps_target - eps_current)

        scene_optimizers[i].zero_grad()
        target = (x0 - grad_x).detach()
        loss_vsd = 0.5 * F.mse_loss(x0, target, reduction="sum")
        loss_vsd.backward()  # Produces the VSD gradient J_g.T @ grad_x.
        scene_optimizers[i].step()

        # Fit the auxiliary denoiser to the same render (Equation 8).
        s = sample_uniform_timestep()  # Uniform over diffusion time.
        aux_noise = torch.randn_like(x_clean)
        x_s = alpha(s) * x_clean + sigma(s) * aux_noise

        aux_optimizer.zero_grad()
        eps_pred = auxiliary(x_s, s, c, y)
        loss_aux = F.mse_loss(eps_pred, aux_noise, reduction="sum")
        loss_aux.backward()
        aux_optimizer.step()

    return particles
```

Each iteration reuses the same rendered image for both updates. `torch.no_grad()` keeps the score estimates fixed during the scene update, while `detach()` prevents the auxiliary loss from updating the scene. The auxiliary initialization creates trainable LoRA adapters while keeping the teacher unchanged. SGD makes the gradient steps explicit here; implementations can use other optimizers. The teacher prediction can include classifier-free guidance.

The code uses two scalar losses. **`loss_vsd` updates the scene parameters through the VSD gradient**, while **`loss_aux` trains the auxiliary denoiser to estimate the current rendered-image score**.

The underlying VSD objective is the weighted KL divergence in Equation 5. The paper derives the particle update in Equation 9 without requiring us to evaluate that KL numerically. To implement this update with a scalar loss and `.backward()`, define, for one rendered image,

$$
d=\omega(t)\left(
\epsilon_{\mathrm{pretrain}}(x_t,t,y^c)
-\epsilon_\phi(x_t,t,c,y)
\right),
\qquad
x_{\mathrm{target}}=\operatorname{stopgrad}(x_0-d).
$$

The scene loss in the code is the surrogate

$$
\ell_{\mathrm{scene}}
=\frac{1}{2}\left\|x_0-x_{\mathrm{target}}\right\|_2^2.
$$

Because the entire target is detached, its derivative is zero during backpropagation. At the current parameters, $$x_0-x_{\mathrm{target}}=d$$, so

$$
\nabla_\theta\ell_{\mathrm{scene}}
=J_g^\top(x_0-x_{\mathrm{target}})
=\omega(t)J_g^\top\left(
\epsilon_{\mathrm{pretrain}}-\epsilon_\phi
\right).
$$

Taking the expectation over cameras, timesteps, and noise recovers the VSD gradient in Equation 9. Thus, `loss_vsd.backward()` is equivalent to the previous `x0.backward(gradient=grad_x)` update. The surrogate's numerical value is not the KL divergence: its purpose is to produce the correct gradient, with a new detached target constructed at each iteration. The factor $$1/2$$ and `reduction="sum"` give the exact scaling shown above for one rendered image.

For the auxiliary update, the loss is ordinary denoising regression:

$$
\ell_{\mathrm{aux}}
=\left\|
\epsilon_\phi\!\left(
\alpha_s\operatorname{stopgrad}(x_0)+\sigma_s\xi,s,c,y
\right)-\xi
\right\|_2^2,
\qquad \xi\sim\mathcal{N}(0,I).
$$

Its expected value is the auxiliary training objective in Equation 8. Learning to predict the added noise supplies the estimate $$-\sigma_s\nabla_{x_s}\log q_s^\mu\approx\epsilon_\phi$$ used in the next VSD updates. The rendered image is detached here so that this loss updates only $$\phi$$. The two losses are optimized alternately over their respective parameters. [ProlificDreamer, Equations 8–9](https://arxiv.org/html/2305.16213v2#S3.SS2)
