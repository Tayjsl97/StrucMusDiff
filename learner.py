import numpy as np
import os
import torch
import json
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Optional
from pathlib import Path
import torch.nn.functional as F
from utils import nested_map

class StrucMusDiffLearner:
    def __init__(
        self, output_dir, model_name, model, train_dl, val_dl, optimizer, params, param_scheduler=None
    ):
        self.output_dir = output_dir
        self.model_name=model_name
        self.log_dir = f"{output_dir}/logs"
        self.checkpoint_dir = f"{output_dir}/chkpts"
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.params = params
        self.param_scheduler = param_scheduler  # teacher-forcing stuff
        self.step = 0
        self.epoch = 0
        self.grad_norm = 0.
        self.summary_writer = None
        self.is_from_scratch = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = torch.cuda.amp.autocast(enabled=params.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=params.fp16)
        self.best_cnt=0

        self.best_val_loss = torch.tensor([1e10], device=self.device)

        # restore if directory exists
        # print("???? output_dir exit !!!!",self.output_dir)
        if os.path.exists(self.output_dir):
            #print(Path(self.output_dir).parent.name[4:],model_name)
            if Path(self.output_dir).parent.name[4:]!=model_name:
                print("Change save path !!!")
                self.restore_from_checkpoint(fname="best")
                output_dir = 'saved_models/ME_'+model_name+'/'
                output_dir = f"{output_dir}/{datetime.now().strftime('%m-%d_%H%M%S')}"
                self.output_dir = output_dir
                self.log_dir = f"{output_dir}/logs"
                self.checkpoint_dir = f"{output_dir}/chkpts"
                os.makedirs(self.output_dir)
                os.makedirs(self.log_dir)
                os.makedirs(self.checkpoint_dir)
            else:
                print("Keep original path !!!")
                self.is_from_scratch=False
                self.restore_from_checkpoint(fname="best")
        else:
            os.makedirs(self.output_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.checkpoint_dir)
        with open(f"{output_dir}/params.json", "w") as params_file:
            json.dump(self.params, params_file)

        # print(json.dumps(self.params, sort_keys=True, indent=4))

    def _write_summary(self, losses: dict, type, is_inpaint):
        """type: train or val"""
        summary_losses = losses
        # summary_losses["grad_norm"] = self.grad_norm
        writer = self.summary_writer or SummaryWriter(
            self.log_dir, purge_step=self.step
        )
        writer.add_scalars('loss/loss', {type:summary_losses['loss']}, self.step)
        # writer.add_scalars(type, summary_losses, self.step)
        if is_inpaint:
            writer.add_scalars('loss/noise_loss1',{type:summary_losses['noise_loss1']},self.step)
            writer.add_scalars('loss/noise_loss2', {type: summary_losses['noise_loss2']}, self.step)
            writer.add_scalars('loss/phrase_loss', {type: summary_losses['phrase_loss']}, self.step)
            writer.add_scalars('loss/str_loss', {type: summary_losses['str_loss']}, self.step)
        writer.flush()
        self.summary_writer = writer

    def state_dict(self):
        model_state = self.model.state_dict()
        return {
            "step": self.step,
            "epoch": self.epoch,
            "model":
                {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in model_state.items()
                },
            "optimizer":
                {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in self.optimizer.state_dict().items()
                },
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if self.is_from_scratch:
            self.step = 0#state_dict["step"]
            self.epoch = 0#state_dict["epoch"]
        else:
            self.step = state_dict["step"]
            self.epoch = state_dict["epoch"]
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])

    def restore_from_checkpoint(self, fname="weights"):
        try:
            fpath = f"{self.checkpoint_dir}/{fname}.pt"
            checkpoint = torch.load(fpath)
            self.load_state_dict(checkpoint)
            print(f"Restored from checkpoint {fpath} --> {fname}-{self.epoch}.pt!")
            return True
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch...")
            return False

    def _link_checkpoint(self, save_name, link_fpath):
        if os.path.islink(link_fpath):
            os.unlink(link_fpath)
        os.symlink(save_name, link_fpath)

    def write_epoch_info(self,loss):
        with open(f"{self.checkpoint_dir}/info.txt", "a") as f:
            if self.best_cnt == 0:
                f.write(str(self.epoch)+" ")
                f.write(str(self.best_val_loss))
                f.write(datetime.now().strftime('%m-%d_%H%M%S'))
                f.write("     best!\n")
            else:
                f.write(str(self.epoch) + " ")
                f.write(str(loss))
                f.write(datetime.now().strftime('%m-%d_%H%M%S')+"\n")
            f.close()

    def save_to_checkpoint(self, fname="weights", is_best=False):
        save_fpath = f"{self.checkpoint_dir}/{fname}.pt"
        save_best_fpath = f"{self.checkpoint_dir}/{fname}_best.pt"
        save_best_fpath_only = f"{self.checkpoint_dir}/best.pt"
        # link_best_fpath = f"{self.checkpoint_dir}/{fname}_best.pt"
        # link_fpath = f"{self.checkpoint_dir}/{fname}.pt"
        if is_best:
            # self._link_checkpoint(save_name, link_best_fpath)
            torch.save(self.state_dict(), save_best_fpath)
            torch.save(self.state_dict(), save_best_fpath_only)
            # self.write_epoch_info(0)
        else:
            torch.save(self.state_dict(), save_fpath)
        # self._link_checkpoint(save_name, link_fpath)


    def train(self, max_epoch=None,is_inpaint=False):
        self.model.train()

        while True:
            if self.best_cnt==20:
                break
            if self.param_scheduler is not None:
                self.param_scheduler.train()
            if max_epoch is not None and self.epoch >= max_epoch:
                return

            for _step, batch in enumerate(
                tqdm(self.train_dl, desc=f"Epoch {self.epoch}")
            ):
                batch = nested_map(
                    batch, lambda x: x.to(self.device)
                    if isinstance(x, torch.Tensor) else x
                )
                losses = self.train_step(batch,is_inpaint)
                # check NaN
                for loss_value in list(losses.values()):
                    if isinstance(loss_value,
                                  torch.Tensor) and torch.isnan(loss_value).any():
                        raise RuntimeError(
                            f"Detected NaN loss at step {self.step}, epoch {self.epoch}"
                        )
                self.step += 1
                if self.step % 100 == 0:
                    self._write_summary(losses, "train",is_inpaint)
                if _step % 5000 == 4999:
                    break
                    # self.model.eval()
                    # self.valid()
                    # self.model.train()
            self.epoch += 1

            # valid
            self.model.eval()
            self.valid(self.epoch,is_inpaint)
            self.model.train()
        return self.output_dir

    def valid(self,epoch,is_inpaint):
        if self.param_scheduler is not None:
            self.param_scheduler.eval()
        losses = None
        for batch in self.val_dl:
            batch = nested_map(
                batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x
            )
            current_losses = self.val_step(batch,is_inpaint)
            if losses is None:
                losses = current_losses
            else:
                for k, v in current_losses.items():
                    losses[k] += v
        assert losses is not None
        for k, v in losses.items():
            losses[k] /= len(self.val_dl)
        self._write_summary(losses,"val",is_inpaint)

        if self.best_val_loss >= losses["loss"]:
            self.best_val_loss = losses["loss"]
            self.best_cnt = 0
            self.save_to_checkpoint(fname=str(epoch),is_best=True)
            self.write_epoch_info(0)
        else:
            self.best_cnt += 1
            self.save_to_checkpoint(fname=str(epoch),is_best=False)
            self.write_epoch_info(losses["loss"])
        if is_inpaint:
            with open(f"{self.checkpoint_dir}/info.txt", "a") as f:
                f.write(f"noise1: {round(losses['noise_loss1'],6)} "
                        f"noise2: {round(losses['noise_loss2'],6)} "
                        f"phrase: {round(losses['phrase_loss'],6)} "
                        f"str: {round(losses['str_loss'],6)}\n")
            f.close()

    def train_step(self, batch,is_inpaint):
        # people say this is the better way to set zero grad
        # instead of self.optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

        # here forward the model
        #with self.autocast:
        loss_dict,_ = self.model.get_loss_dict(batch,is_inpaint)
        loss = loss_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.optimizer.step()
        #self.scaler.scale(loss).backward()
        #self.scaler.unscale_(self.optimizer)
        #self.grad_norm = nn.utils.clip_grad.clip_grad_norm_(
        #    self.model.parameters(), self.params.max_grad_norm or 1e9
        # )
        #self.scaler.step(self.optimizer)
        #self.scaler.update()
        return loss_dict

    def val_step(self, batch,is_inpaint):
        with torch.no_grad():
            #with self.autocast:
            loss_dict,_ = self.model.get_loss_dict(batch,is_inpaint)

        return loss_dict