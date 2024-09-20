import torch

from .ptp.ptp_utils import register_attention_control


class PromptToPromptPipeline:
    def __init__(self, sd):
        self.sd = sd

    # implementation of https://arxiv.org/abs/2310.01506
    @torch.no_grad()
    def getInvOffsets(self, z, prompt, guidance_scale):
        c_pos = self.sd.getEmb([prompt])
        c_neg = self.sd.getEmb([''])
        inv_zs = [z.clone()]
        for t in torch.flip(self.sd.scheduler.timesteps, dims=[0]):
            # guidance scale 0 for inversion
            z = self.sd.step(z, c_pos, c_neg, t, 0, inv=True)
            inv_zs.append(z.clone())
        z_T = z.clone()
        inv_offsets = []
        inv_zs = inv_zs[-2 : : -1]
        for i, t in enumerate(self.sd.scheduler.timesteps):
            z = self.sd.step(
                z, c_pos, c_neg, t, guidance_scale, inv=False)
            inv_offsets.append(inv_zs[i] - z)
            z += inv_offsets[i]
        return z_T, inv_offsets

    @torch.no_grad()
    def recon(self, z, prompt, guidance_scale, inv_offsets):
        c_pos = self.sd.getEmb([prompt])
        c_neg = self.sd.getEmb([''])
        for i, t in enumerate(self.sd.scheduler.timesteps):
            z = self.sd.step(
                z, c_pos, c_neg, t, guidance_scale, inv=False)
            if inv_offsets != None:
                z += inv_offsets[i]
        image = self.sd.latent2Image(z)
        return image

    @torch.no_grad()
    def run(self, z, prompts, controller, guidance_scale, inv_offsets):
        if isinstance(prompts, str):
            prompts = [prompts]
        if controller != None:
            register_attention_control(self.sd, controller)
        c_pos = self.sd.getEmb(prompts)
        c_neg = self.sd.getEmb([''] * len(prompts))
        z = torch.cat([z] * len(prompts), dim=0)
        for i, t in enumerate(self.sd.scheduler.timesteps):
            z = self.sd.step(
                z, c_pos, c_neg, t, guidance_scale, inv=False)
            # XXX: before or after ptp?
            if controller != None:
                z = controller.step_callback(z)
            if inv_offsets != None:
                z += inv_offsets[i]
        image = self.sd.latent2Image(z)
        return image

