import torch
import torch.nn as nn
import sys

from .model_utils import *
from stable_diffusion.latent_diffusion import LatentDiffusion as LatentDiffusion
import torch.nn.functional as F
import random
from models.SSIM_func import SSIM


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class StrucMusDiff(nn.Module):
    def __init__(
        self,
        ldm: LatentDiffusion,
        cond_type,
        cond_mode="cond",
        similar_phrase_enc=None,
    ):
        """
        cond_type: {pp, str}
        cond_mode: {cond, uncond}
            cond: use a special condition for unconditional learning with probability of 0.2
        """
        super(StrucMusDiff, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ldm = ldm
        self.cond_type = cond_type
        self.cond_mode = cond_mode
        self.similar_phrase_enc=similar_phrase_enc
        self.phraseMark_emb=Embeddings(3,64)
        self.simiValue_linear1=nn.Linear(1,16)
        self.simiValue_linear2=nn.Linear(16,64)
        self.cond_linear_str1 = nn.Linear(128 + similar_phrase_enc.width, 128*4)
        self.cond_linear_str2 = nn.Linear(128*4, 128)
        self.cond_linear_pm1 = nn.Linear(64, 64*4)
        self.cond_linear_pm2 = nn.Linear(64*4, 128)

    @classmethod
    def load_trained(
        cls,
        ldm,
        chkpt_fpath,
        cond_type,
        cond_mode="cond",
        similar_phrase_enc=None
    ):
        model = cls(
            ldm, cond_type, cond_mode, similar_phrase_enc
        )
        trained_leaner = torch.load(chkpt_fpath)
        model.load_state_dict(trained_leaner["model"])
        return model

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        return self.ldm.p_sample(xt, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ldm.q_sample(x0, t)

    def _encode_similar_phrase(self, similar_phrase):
        z = self.similar_phrase_enc(similar_phrase)
        return z

    def get_cond(self,similar_phrase, similar_value, phrase_mark, ratio=0.1):
        if self.cond_type.find("str")!=-1:
            pm_cond = self.phraseMark_emb(phrase_mark)
            pm_cond=self.cond_linear_pm2(self.cond_linear_pm1(pm_cond)).unsqueeze(1)
            spv_cond = self.simiValue_linear2(self.simiValue_linear1(similar_value.unsqueeze(-1))).unsqueeze(1)
            if self.similar_phrase_enc is not None:
                spe_cond = self._encode_similar_phrase(similar_phrase)
            else:
                spe_cond = similar_phrase
            pm_cond1 = pm_cond.repeat(1, spe_cond.shape[1], 1)
            spv_cond1 = spv_cond.repeat(1, spe_cond.shape[1], 1)
            str_cond = torch.cat((spe_cond, spv_cond1, pm_cond1), dim=-1)
            cond = self.cond_linear_str2(self.cond_linear_str1(str_cond))
        elif self.cond_type == "pp":
            pm_cond = self.phraseMark_emb(phrase_mark)
            pm_cond=self.cond_linear_pm2(self.cond_linear_pm1(pm_cond))
            cond=pm_cond.repeat((1,similar_phrase.shape[1],1))
        else:
            cond = torch.zeros((similar_phrase.shape[0], similar_phrase.shape[1], 128)).to(device)
        if self.cond_mode == "uncond":
            cond = (-torch.ones((similar_phrase.shape[0], similar_phrase.shape[1], 128))).to(self.device)  # a bunch of -1
        elif self.cond_mode == "cond":
            if random.random() < ratio:
                cond = (-torch.ones((similar_phrase.shape[0], similar_phrase.shape[1], 128))).to(self.device)  # a bunch of -1
        return cond


    def unnormalize(self, x,mean=4.3563,std=22.6885):
        x = x * (std + 1e-8)
        x = x + mean
        return x


    def get_loss_dict(self, batch, is_inpaint):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        input, similar_phrase, similar_index, similar_value, phrase_mark, emotion = batch
        cond = self.get_cond(similar_phrase, similar_value, phrase_mark)
        if is_inpaint:
            noise_loss1,noise_loss2,phrase_loss,x0_gen = self.ldm.loss_inpaint(input.unsqueeze(1), cond)
            ssim = SSIM().to(device)
            sim_v = []
            for i in range(x0_gen.shape[0]):
                img1=similar_phrase.unsqueeze(0).unsqueeze(1).narrow(0,i,1)
                img2=x0_gen.narrow(0,i,1)
                img1 = self.unnormalize(img1)
                img2 = self.unnormalize(img2)
                sim_v.append(ssim(img1, img2))
            sim_v = torch.stack(sim_v)
            str_loss = F.mse_loss(sim_v, similar_value)
            loss=noise_loss2+0.2*str_loss
            return {"loss": loss, "noise_loss1":noise_loss1.item(),"noise_loss2":noise_loss2.item(),
                    "phrase_loss":phrase_loss.item(),"str_loss":str_loss.item()}, cond
        else:
            loss = self.ldm.loss(input, cond)
            return {"loss": loss}, cond
