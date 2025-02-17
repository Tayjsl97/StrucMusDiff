import os.path
from typing import Optional, List
import numpy as np

import numpy as np
import torch

from labml import monit
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler import DiffusionSampler

from utils import show_image
from scheduler import get_schedule_jump


class EMDSampler(DiffusionSampler):
    """
    ## DDPM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDPM samples images by repeatedly removing noise by sampling step by step from
    $p_\theta(x_{t-1} | x_t)$,

    \begin{align}

    p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big) \\

    \mu_t(x_t, t) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\

    \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t \\

    x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta \\

    \end{align}
    """

    model: LatentDiffusion

    def __init__(
        self,
        model: LatentDiffusion,
        is_autocast=False,
        is_show_image=False,
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__(model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Sampling steps $1, 2, \dots, T$
        self.time_steps = np.asarray(list(range(self.n_steps)), dtype=np.int32)

        self.is_show_image = is_show_image

        self.autocast = torch.cuda.amp.autocast(enabled=is_autocast)

        with torch.no_grad():
            # $\bar\alpha_t$
            alpha_bar = self.model.alpha_bar
            # $\beta_t$ schedule
            beta = self.model.beta
            #  $\bar\alpha_{t-1}$
            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])

            # $\sqrt{\bar\alpha}$
            self.sqrt_alpha_bar = alpha_bar**.5
            # $\sqrt{1 - \bar\alpha}$
            self.sqrt_1m_alpha_bar = (1. - alpha_bar)**.5
            # $\frac{1}{\sqrt{\bar\alpha_t}}$
            self.sqrt_recip_alpha_bar = alpha_bar**-.5
            # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
            self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1)**.5

            # $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$
            variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
            # Clamped log of $\tilde\beta_t$
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
            self.mean_x0_coef = beta * (alpha_bar_prev**.5) / (1. - alpha_bar)
            # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
            self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta)**
                                                         0.5) / (1. - alpha_bar)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        step: int,
        repeat_noise: bool = False,
        temperature: float = 1.,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None
    ):
        """
        ### Sample $x_{t-1}$ from $p_\theta(x_{t-1} | x_t)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $t$ of shape `[batch_size]`
        :param step: is the step $t$ as an integer
        :repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # Get $\epsilon_\theta$
        with self.autocast:
            e_t = self.get_eps(
                x, t, c, uncond_scale=uncond_scale, uncond_cond=uncond_cond
            )

        # Get batch size
        bs = x.shape[0]

        # $\frac{1}{\sqrt{\bar\alpha_t}}$
        sqrt_recip_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step]
        )
        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        sqrt_recip_m1_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step]
        )

        # Calculate $x_0$ with current $\epsilon_\theta$
        #
        # $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta$$
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        # Calculate $\mu_t(x_t, t)$
        #
        # $$\mu_t(x_t, t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #    + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        # $\log \tilde\beta_t$
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if step == 0:
            noise = 0
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1 :]), device=self.device)
        # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=self.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        # Sample from,
        #
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        x_prev = mean + (0.5 * log_var).exp() * noise

        #
        return x_prev, x0, e_t

    @torch.no_grad()
    def q_sample(
        self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None
    ):
        """
        ### Sample from $q(x_t|x_0)$

        $$q(x_t|x_0) = \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)
        # print("noise: ",noise)
        # Sample from $\mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
        t_start: int = 0,
        is_show: bool=True
    ):
        """
        ### Sampling Loop

        :param shape: is the shape of the gen_midi images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $t'$. We start sampling from $T - t'$.
            And `x_last` is then $x_{T - t'}$.
        """

        # Get device and batch size
        bs = shape[0]

        # Get $x_T$
        x = x_last if x_last is not None else torch.randn(shape, device=self.device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(self.time_steps)[t_start :]

        # Sampling loop
        for step in monit.iterate('Sample', time_steps):
            # Time step $t$
            ts = x.new_full((bs, ), step, dtype=torch.long)

            # Sample $x_{t-1}$
            x, pred_x0, e_t = self.p_sample(
                x,
                cond,
                ts,
                step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond
            )

            s1 = step + 1
            if is_show:
                if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                    show_image(x, f"exp/img/x{s1}.png")

        # Return $x_0$
        if is_show:
            show_image(x, f"exp/img/x0.png")
        return x

    @torch.no_grad()
    def paint(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_start: int,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
        model_label: str="",
        is_show: bool=True,
        img_path: str="",
        seg_num:int=-1,
        repaint: bool=False
    ):
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # Get  batch size
        bs = x.shape[0]
        if not repaint:
            # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
            time_steps = np.flip(self.time_steps[: t_start])

            for i, step in monit.enum('Paint', time_steps):
                # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
                # index = len(time_steps) - i - 1
                # Time step $\tau_i$
                ts = x.new_full((bs, ), step, dtype=torch.long)
                # print(i,step)
                # print("p_sample: ",x.shape,cond.shape,ts.shape)
                # Sample $x_{\tau_{i-1}}$
                x, _, _ = self.p_sample(
                    x,
                    cond,
                    ts,
                    step,
                    uncond_scale=uncond_scale,
                    uncond_cond=uncond_cond
                )

                # Replace the masked area with original image
                if torch.sum(mask)!=0:
                    # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                    # orig_t = self.q_sample(orig, step, noise=orig_noise)
                    orig_t = self.q_sample(orig, step)
                    # Replace the masked area
                    x = orig_t * mask + x * (1 - mask)

                s1 = step + 1
                if is_show:
                    # if s1 % 100 == 0 or (s1 >= 900 and s1 % 10 == 0) or s1 >= 990:
                    if s1 % 100 == 0:
                        if seg_num==-1:
                            show_image(x, f"{img_path}/x{s1}.png")
                        else:
                            show_image(x, f"{img_path}/{seg_num}_x{s1}.png")
        else:
            times = get_schedule_jump(
                t_T=1000,
                n_sample=1,
                jump_length=200,
                jump_n_sample=10,
                start_resampling=600,
                end_resampling=200
            )
            time_pairs = list(zip(times[:-1], times[1:]))
            # for t_last, t_cur in time_pairs:
            for i,(t_last, t_cur) in monit.enum('Paint', time_pairs):
                if t_cur==-1:
                    break
                if t_cur<t_last:   # denoise
                    ts = x.new_full((bs,), t_cur, dtype=torch.long)
                    # Sample $x_{\tau_{i-1}}$
                    x, _, _ = self.p_sample(
                        x,
                        cond,
                        ts,
                        t_cur,
                        uncond_scale=uncond_scale,
                        uncond_cond=uncond_cond
                    )

                    # Replace the masked area with original image
                    if torch.sum(mask) != 0:
                        # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                        # orig_t = self.q_sample(orig, t_cur, noise=orig_noise)
                        orig_t = self.q_sample(orig, t_cur)
                        # Replace the masked area
                        x = orig_t * mask + x * (1 - mask)
                    s1 = t_cur + 1
                    if is_show:
                        # if s1 % 100 == 0 or (s1 >= 900 and s1 % 10 == 0) or s1 >= 990:
                        if s1 % 100 == 0:
                            if seg_num == -1:
                                show_image(x, f"{img_path}/x{s1}.png")
                            else:
                                show_image(x, f"{img_path}/{seg_num}_x{s1}.png")
                else:
                    noise = torch.randn_like(x, device=self.device)
                    x = ((1-self.model.beta[t_last])**.5) * x +\
                        (self.model.beta[t_last]**.5) * noise
        if is_show:
            if seg_num == -1:
                show_image(x, f"{img_path}/x0.png")
            else:
                show_image(x, f"{img_path}/{seg_num}_x0.png")
        return x
