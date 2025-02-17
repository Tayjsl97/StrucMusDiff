"""
---
title: Denoising Diffusion Probabilistic Models (DDPM) Sampling
summary: >
 Annotated PyTorch implementation/tutorial of
 Denoising Diffusion Probabilistic Models (DDPM) Sampling
 for stable diffusion model.
---

# Denoising Diffusion Probabilistic Models (DDPM) Sampling

For a simpler DDPM implementation refer to our [DDPM implementation](../../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""
import os.path
from typing import Optional, List
from pathlib import Path
import torch.nn.functional as F
import json
import random,copy
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.model.unet import UNetModel
from models.StrucMusDiff import StrucMusDiff
from models.Reference_Phrase_Encoder import Reference_Phrase_Encoder
from params import AttrDict

from argparse import ArgumentParser
import pickle
from datetime import datetime
from utils import *
from sampler import EMDSampler
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = ArgumentParser(description='inference a Polyffusion model')
torch.set_printoptions(threshold=np.inf)

def get_mask(orig):
    mask = torch.ones_like(orig)
    index=mask.shape[2]//2
    mask[:,:, index:, :] = 0
    return mask


class Experiments:
    def __init__(self, model_label, params, sampler: EMDSampler) -> None:
        self.model_label = model_label
        self.params = params
        self.sampler = sampler

    def predict(
        self,
        cond: torch.Tensor,
        uncond_scale=1.,
        orig=None,
        mask=None,
        is_show=True,
        seg_num=-1,
        repaint=False
    ):
        uncond_cond = (-torch.ones_like(cond)).to(device)
        shape = [cond.shape[0], self.params.out_channels, self.params.img_h, self.params.img_w]
        # a bunch of -1
        #print("what: ",orig.shape,cond.shape)
        # print(f"generating {shape} with uncond_scale = {uncond_scale}")
        self.sampler.model.eval()
        if orig is None:
            orig = torch.zeros(shape, device=device)
        if mask is None:
            mask = torch.zeros(shape, device=device)
        t_idx = self.params.n_steps - 1
        #t_idx = sampler.real_steps - 1
        noise = torch.randn(shape, device=device)
        with torch.no_grad():
            xt = self.sampler.q_sample(orig, t_idx, noise)
            gen = self.sampler.paint(
                xt,
                cond,
                t_idx,
                orig=orig,
                mask=mask,
                orig_noise=noise,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
                model_label=model_label,
                is_show=is_show,
                img_path=img_path,
                seg_num=seg_num,
                repaint=repaint
            )
        if seg_num==-1:
            show_image(gen, f"{img_path}/gen.png")
        else:
            show_image(gen, f"{img_path}/{seg_num}_gen.png")
        return gen

    def generate(
        self,
        orig: torch.Tensor,
        cond: torch.Tensor,
        uncond_scale=1.,
        autoreg=False,
        no_output=False,
        repaint=False
    ):
        if orig is not None:
            gen = self.predict(
                cond,
                uncond_scale,
                orig=orig,
                seg_num=0,
                repaint=repaint
            )
        else:
            gen = self.predict(
                cond,
                uncond_scale,
                seg_num=0,
                repaint=repaint
            )
        return gen

    def inpaint(
        self,
        orig: torch.Tensor,
        cond: torch.Tensor,
        orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        is_show=True,
        seg_num=-1,
        repaint=False
    ):
        # show_image(orig, "exp/img/orig.png")
        orig_noise = orig_noise or torch.randn(orig.shape, device=device)
        mask = get_mask(orig)
        # show_image(mask.bool(), "img/mask.png", cmap="Greys")
        mask = mask.to(device)
        gen = self.predict(
            cond, uncond_scale, orig, mask, is_show,seg_num,repaint=repaint
        )
        return gen

    def show_q_imgs(self, prmat2c,val_idx):
        q_img_path=f"img/{model_label}/inpaint/{val_idx}/q"
        if not os.path.exists(q_img_path):
            os.makedirs(q_img_path)
        show_image(prmat2c, f"{q_img_path}/q0.png")
        for step in self.sampler.time_steps:
            s1 = step + 1
            if s1 % 100 == 0:
                noised = self.sampler.q_sample(prmat2c, step)
                show_image(noised, f"{q_img_path}/q{s1}.png")



if __name__ == "__main__":
    # f = open('logs/help.log', 'a')
    parser.add_argument(
        "--model_dir", help='directory in which trained model checkpoints are stored'
    )
    parser.add_argument("--params_dir", help='directory in which params are stored')
    parser.add_argument(
        "--uncond_scale",
        default=0.,
        help="unconditional scale for classifier-free guidance. "
             "0 for unconditional generation, >0 for conditional generation."
    )
    parser.add_argument("--seed", help="use a specific seed for inference")
    parser.add_argument(
        "--save_midi", default="gen_midi",help="choose condition from a specific midi file"
    )
    parser.add_argument("--length", default=1, help="the gen_midi length (in 8-bars)")
    parser.add_argument("--task_type", default="inpaint", help="task type: inpaint, inpaint_repaint, generate_from_noise,"
                                                                  "generate_from_exist, generate (via inpaint)")
    parser.add_argument("--val_idx", default=0, help="the index pf validation sample")
    parser.add_argument("--val_num", default=0, help="the number pf validation sample")
    parser.add_argument("--sim_value", default=0.9, help="the similarity score")
    parser.add_argument("--phrase_mark", default=-1, help="the phrase mark condition")
    # you usually don't need to use the following args
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="whether to show the images of gen_midi piano-roll"
    )
    parser.add_argument(
        "--chkpt_name",
        default="best.pt",
        help="which specific checkpoint to use (default: weights_best.pt)"
    )
    parser.add_argument(
        "--only_q_imgs",
        action="store_true",
        help="only show q_sample results (for testing)"
    )
    args = parser.parse_args()
    model_label = Path(args.model_dir).parent.name+"/"+Path(args.model_dir).name
    print(f"model_label: {model_label}")

    if args.seed is not None:
        SEED = int(args.seed)
        print(f"fixed SEED = {SEED}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    # params ready
    with open(f"{args.model_dir}/params.json", "r") as params_file:
        params = json.load(params_file)
    params = AttrDict(params)
    print("params: ",params)

    # model ready
    autoencoder = None
    unet_model = UNetModel(
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        channels=params.channels,
        attention_levels=params.attention_levels,
        n_res_blocks=params.n_res_blocks,
        channel_multipliers=params.channel_multipliers,
        n_heads=params.n_heads,
        tf_layers=params.tf_layers,
        d_cond=params.d_cond
    )

    ldm_model = LatentDiffusion(
        linear_start=params.linear_start,
        linear_end=params.linear_end,
        n_steps=params.n_steps,
        # latent_scaling_factor=params.latent_scaling_factor,
        autoencoder=autoencoder,
        unet_model=unet_model
    )

    model = StrucMusDiff.load_trained(
        ldm_model,
        f"{args.model_dir}/chkpts/best.pt",
        params.cond_type,
        params.cond_mode,
        similar_phrase_enc=Reference_Phrase_Encoder(params.SPE_layers, params.SPE_heads, params.width,
                                                  params.dropout, params.activate, params.attention_type)
    ).to(device)
    model.eval()
    sampler = EMDSampler(
        model.ldm,
        is_autocast=params.fp16,
        is_show_image=args.show_image,
    )
    expmt = Experiments(model_label, params, sampler)
    gen_record = np.full(3, 80)
    # gen_record = 80
    img_root_path = f"img/{model_label}/{args.task_type}_{args.uncond_scale}/"
    already_idx=set()
    # val_dl = list(val_dl)[4691:]
    file = open(params.data_path, 'rb')
    data = pickle.load(file)
    input_data_all = torch.Tensor(data['input']).to(device)
    simi_data_all = torch.Tensor(data['sim_data']).to(device)
    pm_all = torch.LongTensor(data['phrase_mark']).to(device)
    similar_value_all=torch.Tensor(data['sim_value']).to(device)
    #simi_data_all = torch.where((simi_data_all > 0) & (simi_data_all < 128), 128, simi_data_all).float()
    # for val_idx in range(int(args.val_idx),int(args.val_idx)+int(args.val_num)):
    cnt=0
    indices = torch.randperm(input_data_all.size(0)).to(device)
    input_data_all = torch.index_select(input_data_all, 0, indices)
    simi_data_all = torch.index_select(simi_data_all, 0, indices)
    similar_value_all = torch.index_select(similar_value_all, 0, indices)
    pm_all = torch.index_select(pm_all, 0, indices)
    sim_metrics_per_song=[]
    # inpaint
    for val_idx in range(50000):
        if np.sum(gen_record)==0:
            break
        print(f"\nBegin inpainting/generating the {cnt} example---------------")
        input_data=normalize(input_data_all[val_idx]).unsqueeze(0)
        similar_phrase=normalize(simi_data_all[val_idx]).unsqueeze(0)
        similar_value=similar_value_all[val_idx].unsqueeze(0)
        phrase_mark=pm_all[val_idx].item()
        midi_path_GT = os.path.join(args.save_midi, model_label, args.task_type) + "_GT"
        if not os.path.exists(midi_path_GT):
            os.makedirs(midi_path_GT)
        print("----------Given condition---------")
        if gen_record[phrase_mark] > 0:
            output_stamp_GT = f"{cnt}_[pm={phrase_mark}]_GT"
            pr_to_midi_file(input_data.cpu().numpy(), f"{midi_path_GT}/{output_stamp_GT}.mid")

            phrase_mark_cond = torch.LongTensor([phrase_mark]).to(device)
            model.cond_type = "str"
            cond2 = model.get_cond(similar_phrase, similar_value, phrase_mark_cond, ratio=0)
            print("Structure: \n", "  phrase_mark: ", phrase_mark, "similar value: ", similar_value.item())
        else:
            continue
        if args.task_type != "generate":
            sim_value_tmp=round(similar_value[0].item(),4)
            img_path = img_root_path + f"{cnt}_[pp={phrase_mark}]_[scale={args.uncond_scale}]_[sim_v={sim_value_tmp}]_{datetime.now().strftime('%m-%d_%H%M%S')}"
            print("img_path: ", img_path)
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            show_image(input_data, f"{img_path}/origin.png")
        if args.task_type == "generate_from_noise":
            print("!!! initial input data as noise")
            input_data = torch.randn((1,128,128)).to(device)
        if args.only_q_imgs:
            print("show_q_imgs!!!")
            expmt.show_q_imgs(input_data,val_idx)

        if float(args.uncond_scale) == 0.:
            print("The model is trained unconditionally, ignoring conditions...")
            cond1 = -torch.ones_like(cond1).to(device)
            cond2 = -torch.ones_like(cond1).to(device)


        # generated midi path
        midi_path = os.path.join(args.save_midi, model_label, args.task_type)+"_"+args.uncond_scale
        if not os.path.exists(midi_path):
            os.makedirs(midi_path)

        if args.task_type.find("repaint")!=-1:
            repaint=True
        else:
            repaint=False

        # generate!
        if args.task_type.find("inpaint")!=-1:
            # inpaint!
            print("Task: inpainting")
            output_stamp = f"{cnt}_[pp={phrase_mark}]_[scale={args.uncond_scale}]" \
                           f"_{datetime.now().strftime('%m-%d_%H%M%S')}"
            if float(args.uncond_scale) != 0. and model.cond_type.find("str")!=-1:
                show_image(similar_phrase, f"{img_path}/similar_phrase_{[round(similar_value.item(),4)]}.png")
            input_data_1 = input_data[:,:64,:].unsqueeze(1).to(device)
            input_data_2 = torch.randn((1, 1, 64, 128)).to(device)
            input_data = torch.cat((input_data_1,input_data_2),dim=2)
            gen=expmt.inpaint(
                orig=input_data,  # inapint from noise
                cond=cond2,
                orig_noise=None,
                uncond_scale=float(args.uncond_scale),
                # is_show=False
                repaint=repaint
            )
            pr = gen.cpu().numpy()
            phrase_list=pr_to_midi_file(copy.deepcopy(pr), f"{midi_path}/{output_stamp}.mid")
            gen_record[phrase_mark] -= 1
            cnt+=1
        else:
            print("Task: generating")
            if args.task_type.find("from")!=-1:
                output_stamp = f"{cnt}_[pp={phrase_mark}]_[scale={args.uncond_scale}]" \
                               f"_{datetime.now().strftime('%m-%d_%H%M%S')}"
                gen=expmt.generate(
                            orig=input_data.unsqueeze(1),
                            cond=cond2,
                            uncond_scale=float(args.uncond_scale),
                            no_output=True,
                            repaint=repaint
                    )
                show_image(gen, f"{img_path}/gen.png")
                pr = gen.cpu().numpy()
                pr_to_midi_file(copy.deepcopy(pr), f"{midi_path}/{output_stamp}.mid")
                gen_record[phrase_mark] -= 1
                cnt+=1
            else:
                sim_metrics=[]
                img_path = img_root_path + f"{cnt}_[scale={args.uncond_scale}]_[sim_v={args.sim_value}]_{datetime.now().strftime('%m-%d_%H%M%S')}"
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                show_image(input_data, f"{img_path}/origin.png")
                phrase_list=[]
                for i in range(int(args.length)):
                    if i==0:
                        phrase_mark = torch.LongTensor([0]).to(device)
                        similar_value=torch.Tensor([float(args.sim_value)]).to(device)
                        phrase_mark_cond = torch.LongTensor([phrase_mark]).to(device)
                        model.cond_type = "str"
                        cond2 = model.get_cond(similar_phrase, similar_value, phrase_mark_cond, ratio=0)
                        if float(args.uncond_scale) != 0. and model.cond_type.find("str") != -1:
                            show_image(similar_phrase, f"{img_path}/similar_phrase_{[round(similar_value.item(),4)]}.png")
                        input_data_1 = torch.zeros((1, 1, 64, 128)).to(device)
                        input_data_2 = torch.randn((1, 1, 64, 128)).to(device)
                        input_data = torch.cat((input_data_1,input_data_2),dim=2)
                        #input_data=input_data.unsqueeze(1)
                        #print(input_data.shape,cond2.shape)
                        gen_inp = expmt.inpaint(
                            orig=input_data,
                            cond=cond2,
                            orig_noise=None,
                            uncond_scale=float(args.uncond_scale),
                            seg_num=i,
                            repaint=repaint
                            # is_show=False
                        )
                        gen=gen_inp[:,:,-64:,:]
                        phrase_list.append(gen.cpu().numpy())
                    else:
                        if i==int(args.length)-1:
                            phrase_mark = torch.LongTensor([2]).to(device)
                        else:
                            phrase_mark = torch.LongTensor([1]).to(device)
                        similar_phrase=gen[:,:,-64:,:].squeeze(1)
                        similar_value=torch.Tensor([float(args.sim_value)]).to(device)
                        phrase_mark_cond = torch.LongTensor([phrase_mark]).to(device)
                        show_image(similar_phrase, f"{img_path}/{str(i)}_similar_phrase_{[round(similar_value.item(),4)]}.png")
                        model.cond_type = "str"
                        cond2 = model.get_cond(similar_phrase, similar_value, phrase_mark_cond, ratio=0)
                        input_data=torch.randn_like(gen[:,:,-64:,:]).to(device)
                        input_data=torch.cat((gen[:,:,-64:,:],input_data),dim=2)
                        gen_inp=expmt.inpaint(
                                    orig=input_data,
                                    cond=cond2,
                                    orig_noise=None,
                                    uncond_scale=float(args.uncond_scale),
                                    seg_num=i,
                                    repaint=repaint
                                    # is_show=False
                                )
                        phrase_list.append(gen_inp[:,:,-64:,:].cpu().numpy())
                        gen=torch.cat((gen,gen_inp[:,:,-64:,:]),dim=2)
                    sim_metrics.append(
                        compute_sim_metrics(
                            gen_inp[0,0,-64:,:].cpu().numpy(),
                            similar_phrase[0,:,:].cpu().numpy(),
                            similar_value.item()
                        )
                    )
                show_image(gen, f"{img_path}/gen.png")
                pr = gen.cpu().numpy()
                output_stamp = f"{cnt}_[length={args.length}]_[scale={args.uncond_scale}]" \
                               f"_[sim_v={args.sim_value}]_{datetime.now().strftime('%m-%d_%H%M%S')}"
                phrase_list = pr_to_midi_file(copy.deepcopy(pr), f"{midi_path}/{output_stamp}.mid")
                gen_record-=1
                cnt+=1
                print(sim_metrics)
                sim_metric_value=sim_metrics_RMSE(sim_metrics)
                print("sim_RMSE: ",round(sim_metric_value,6))
                sim_metrics_per_song.append(round(sim_metric_value,6))
        print("sim_metrics_list: ",sim_metrics_per_song)
    print("sim_metrics: ",sum(sim_metrics_per_song)/len(sim_metrics_per_song))
