---
layout: post
categories: [diffusion-models]
title: "4. Improved Distribution Matching Distillation"
date: 2026-11-11 12:00:00 +0100
description: "How DMD2 stabilizes distribution matching, adds GAN training, and prepares a few-step student for inference."
math: true
---

[DMD2: Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867)

DMD2 is a follow-up to DMD by Tianwei Yin and collaborators. I find this progression inspiring: it shows how a research idea can become stronger when its authors follow the same line of work, examine the limitations of their first method, and push it further step by step.

The original DMD trains a one-step generator with a distribution-matching update and an additional regression loss. DMD2 asks whether that regression loss is actually necessary, and what becomes possible when we remove it. The authors stabilize training with a **two-time-scale update rule**, introduce a **GAN loss** using real images, and extend the method to **multi-step generation** with a procedure called backward simulation.

## Why Remove the Regression Loss?

Recall that DMD uses two score estimates: a frozen pretrained diffusion model estimates the target score, while a fake diffusion model learns the score of the student's current output distribution. The difference between them supplies an approximate distribution-matching gradient for the generator.

In practice, the original DMD also uses a regression loss on precomputed pairs $$(z,y)$$, where $$z$$ is the initial noise and $$y$$ is the teacher's output from a deterministic multi-step sampler:

$$
\mathcal L_{\mathrm{reg}}
=\mathbb E_{(z,y)}\left[d\bigl(G_\theta(z),y\bigr)\right].
$$

This loss helps stabilize training and encourages the student to cover modes represented by the teacher's outputs. But it also introduces two limitations.

First, generating a large paired dataset can be expensive. For example, the DMD2 paper estimates that constructing pairs for 12 million prompts with SDXL would take roughly 700 A100-days. Second, the loss encourages the student to follow the teacher's particular noise-to-image mapping, while the distribution-matching objective only asks it to match a distribution. The paired term can therefore restrict the student's freedom to improve on the teacher's sampling paths. This is a practical constraint, not a proof that a student with any nonzero regression weight can never outperform its teacher. [DMD2, Section 3](https://arxiv.org/html/2405.14867v2#S3)

## Stabilizing Training Without Regression

Simply removing the regression loss from DMD causes unstable training and poorer samples. The authors attribute much of this instability to an inaccurate **fake-score model**, $$\mu_{\mathrm{fake}}$$.

The challenge is that the fake distribution changes whenever the generator changes. The fake denoiser is trained on samples from that moving distribution, so it may lag behind. Its score then gives a biased direction to the generator, which changes the distribution again. Without the regression term to regularize training, this feedback can lead to fluctuations rather than convergence.

DMD2 addresses this with a **two-time-scale update rule**: update the fake-score model more frequently than the generator, giving it more opportunities to track the current student distribution. In the paper's experiments, five fake-score updates per generator update provide a useful balance between stability and computational cost. This does not introduce another generator: $$G_\theta$$ remains the student, and $$\mu_{\mathrm{fake}}$$ remains the auxiliary denoiser. [DMD2, Section 4.2](https://arxiv.org/html/2405.14867v2#S4.SS2)

For example, in a noise-prediction parameterization, the fake denoiser can be trained with a standard denoising objective:

$$
\mathcal L_{\mathrm{fake}}(\phi)
=\mathbb E_{z,t,\epsilon}\!\left[
\lambda(t)\left\|
\epsilon_\phi\bigl(\alpha_t\operatorname{stopgrad}(G_\theta(z))
+\sigma_t\epsilon,t\bigr)-\epsilon
\right\|_2^2\right],
\qquad \epsilon\sim\mathcal N(0,I).
$$

The student sample is detached for this update. The goal is to fit the fake denoiser to the generator's current outputs, not to train the generator through the denoising loss. For conditional generation, the appropriate condition is also supplied to the models. The paper writes its diffusion model using a clean-image prediction parameterization; the loss weight $$\lambda(t)$$ depends on the chosen parameterization.

## Adding a GAN Loss

> **Core insight:** The GAN gives the student feedback from real images, which can help correct errors in the teacher's score. Its discriminator shares features with the fake-score model, but they have different jobs: the discriminator predicts real or fake, while the denoiser estimates the student's score for distribution matching.

The frozen teacher provides an approximate real score, so the student can inherit some of its errors. DMD2 adds a GAN loss to give the student another learning signal, this time from real images.

A discriminator $$D$$ learns to distinguish noised real images from noised student-generated images. The student learns to make its generated images look real to the discriminator. If $$x_t^{\mathrm{real}}$$ and $$x_t^{\mathrm{fake}}$$ denote those noised inputs, the standard non-saturating GAN losses can be written as

$$
\mathcal L_D
=-\mathbb E\left[\log D(x_t^{\mathrm{real}})\right]
-\mathbb E\left[\log\left(1-D(x_t^{\mathrm{fake}})\right)\right],
$$

$$
\mathcal L_{\mathrm{GAN},G}
=-\mathbb E\left[\log D(x_t^{\mathrm{fake}})\right].
$$

The first loss trains the discriminator to recognize real and fake samples; the second trains the generator to make its samples look real. For the discriminator update, the student output is detached. For the generator update, the discriminator's parameters stay fixed, while gradients pass through it to the student. The equations above show the standard non-saturating form; the paper's displayed Equation 4 differs in its discriminator fake term. [DMD2, Section 4.3](https://arxiv.org/html/2405.14867v2#S4.SS3)

In DMD2, the discriminator uses a classification head attached to the fake diffusion model's features. The fake model therefore helps with both score estimation and real-versus-fake classification. The student combines the GAN gradient with the original DMD score-difference update:

$$
\nabla_\theta\mathcal L_G
\;\approx\;
\widehat{\nabla_\theta\mathcal L_{\mathrm{DMD}}}
+\lambda_{\mathrm{GAN}}\nabla_\theta\mathcal L_{\mathrm{GAN},G}.
$$

This additional real-data signal can help the student improve beyond a particular teacher and sampler, as reported in the paper's experiments. It does not guarantee improvement on every quality metric. [DMD2, Appendix F.1](https://arxiv.org/html/2405.14867v2#A6.SS1)

## From One Step to Multiple Steps

A single pass may be too restrictive for a complex high-resolution generator such as SDXL. DMD2 therefore also trains a student that can be called a small number of times. At each selected timestep, it predicts a clean sample; except after the final step, that prediction is noised again before the next call.

For a decreasing schedule $$t_1>t_2>\cdots>t_N$$, the process is

$$
\hat{x}^{(i)}_0=G_\theta(x_{t_i},t_i),
\qquad
x_{t_{i+1}}=\alpha_{t_{i+1}}\hat{x}^{(i)}_0
+\sigma_{t_{i+1}}\epsilon_i,
\quad \epsilon_i\sim\mathcal N(0,I),
$$

for $$i=1,\ldots,N-1$$. Starting from Gaussian noise at $$t_1$$, the final prediction $$\hat{x}^{(N)}_0$$ is the output. Thus the student makes $$N$$ generator evaluations; it does not run the teacher's long trajectory. The paper's four-step example uses timesteps $$999,749,499,249$$ on a 1,000-step teacher schedule. [DMD2, Section 4.4](https://arxiv.org/html/2405.14867v2#S4.SS4)

## Training–Inference Mismatch and Backward Simulation

I first encountered this idea through Self Forcing, and it took me some time to understand why it also matters for DMD2. The key is to look at what a multi-step student receives as input at each step.

Imagine a two-step generator:

1. The student starts from random noise and predicts a clean image. This prediction may still contain mistakes.
2. We add noise to that prediction and give it back to the student, which makes a second prediction.

At inference, the second step must work with the **student's own imperfect prediction**. But if we train that step only by adding noise to real images, it sees different inputs during training: those images do not contain mistakes from the student's first step. This difference is the training–inference mismatch.

DMD2 addresses it with **backward simulation**. During training, it runs the current student through the earlier steps, adding noise between predictions just as it does at inference. The resulting noisy student prediction becomes the input for the selected training step. The output of that step is then trained using the distribution-matching and GAN signals.

For the two-step example, this means the second step learns from a noised version of the student's first prediction, rather than only from a noised real image. As the student changes, these training inputs change too, but they follow the same sampling procedure used at inference. A one-step student has no earlier student prediction, so this particular mismatch arises in the multi-step setting. [DMD2, Section 4.5](https://arxiv.org/html/2405.14867v2#S4.SS5)

## DMD2 Pseudocode

This PyTorch-style pseudocode shows the loss calculations and the alternating updates. It assumes the models, optimizers, real-data loader, and noise schedule already exist:

- `student(x, t, condition)` predicts a clean image.
- `teacher(x, t, condition)` and `fake_model.predict_noise(x, t, condition)` predict the added noise.
- `fake_model.discriminate(x, t, condition)` returns a real/fake logit. Its classifier shares features with the fake denoiser.
- `alpha[t]` and `sigma[t]` are scalar noise-schedule coefficients; `student_steps` is a decreasing list of timesteps.

```python
import torch
import torch.nn.functional as F

teacher.requires_grad_(False)

for iteration, (real, condition) in enumerate(real_loader):
    # Select a student step, then generate its input as at inference.
    step = torch.randint(len(student_steps), ()).item()
    with torch.no_grad():
        x = torch.randn_like(real)
        for i in range(step):
            clean_estimate = student(x, student_steps[i], condition)
            t_next = student_steps[i + 1]
            x = alpha[t_next] * clean_estimate + sigma[t_next] * torch.randn_like(x)

    # Update the student once for every five fake-model updates.
    if iteration % 5 == 0:
        fake_model.requires_grad_(False)
        fake = student(x, student_steps[step], condition)

        # DMD: compare the two noise predictions on the same noisy image.
        t = torch.randint(1, len(alpha) - 1, ()).item()
        with torch.no_grad():
            noisy_fake = alpha[t] * fake + sigma[t] * torch.randn_like(fake)
            eps_real = teacher(noisy_fake, t, condition)
            eps_fake = fake_model.predict_noise(noisy_fake, t, condition)
            grad_x = (alpha[t] / sigma[t]) * (eps_real - eps_fake)
            # An implementation may also apply DMD's normalization weight.

        # This surrogate passes grad_x through the student.
        loss_dmd = (fake * grad_x).flatten(1).sum(1).mean()

        # GAN generator loss: make the discriminator classify fake as real.
        gan_t = torch.randint(len(alpha), ()).item()
        noisy_fake = alpha[gan_t] * fake + sigma[gan_t] * torch.randn_like(fake)
        fake_logits = fake_model.discriminate(noisy_fake, gan_t, condition)
        loss_gan_g = F.softplus(-fake_logits).mean()

        optimizer_student.zero_grad()
        (loss_dmd + lambda_gan * loss_gan_g).backward()
        optimizer_student.step()
        fake_model.requires_grad_(True)

    # Generate a detached sample for the fake model and discriminator.
    with torch.no_grad():
        fake = student(x, student_steps[step], condition)

    # Fake denoiser loss: predict the noise added to the student output.
    t = torch.randint(len(alpha), ()).item()
    noise = torch.randn_like(fake)
    noisy_fake = alpha[t] * fake + sigma[t] * noise
    predicted_noise = fake_model.predict_noise(noisy_fake, t, condition)
    loss_denoise = F.mse_loss(predicted_noise, noise)

    # Discriminator loss: classify real as real and fake as fake.
    gan_t = torch.randint(len(alpha), ()).item()
    noisy_real = alpha[gan_t] * real + sigma[gan_t] * torch.randn_like(real)
    noisy_fake = alpha[gan_t] * fake + sigma[gan_t] * torch.randn_like(fake)
    real_logits = fake_model.discriminate(noisy_real, gan_t, condition)
    fake_logits = fake_model.discriminate(noisy_fake, gan_t, condition)
    loss_disc = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()

    optimizer_fake.zero_grad()
    (loss_denoise + lambda_disc * loss_disc).backward()
    optimizer_fake.step()
```

The first loop is **backward simulation**: it produces the selected step's input using earlier student predictions. The student is then updated with the DMD score difference and the GAN generator loss. `softplus(-fake_logits)` encourages positive logits, meaning the discriminator considers the generated images real.

Every iteration also updates the fake model. Its denoising loss teaches it to track the student distribution; its discriminator loss encourages positive logits for real images and negative logits for fake ones. The student output is detached for both updates. During the student update, the fake model's parameters are frozen, but the GAN gradient still flows through it to the student.

This is an illustrative algorithm, not a complete reproduction. It omits classifier-free guidance and the sample-dependent DMD normalization, uses an unweighted denoising loss and one timestep per batch, and requires a schedule where the selected DMD timesteps have nonzero `alpha` and `sigma`. The DMD surrogate's value is not the KL divergence. In latent diffusion, the images and generated samples in this example are represented in latent space. [DMD2, Sections 4.2–4.6](https://arxiv.org/html/2405.14867v2#S4)
