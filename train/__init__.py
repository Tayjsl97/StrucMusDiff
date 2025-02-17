import torch
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import sys

from params import AttrDict
from learner import StrucMusDiffLearner


class TrainConfig():
    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: Optimizer

    def __init__(self, params, output_dir, is_inpaint=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.output_dir = output_dir
        self.is_inpaint = is_inpaint
        # if os.path.exists(f"{output_dir}/params.json"):
        #     with open(f"{output_dir}/params.json", "r") as params_file:
        #         self.params = AttrDict(json.load(params_file))
        #         if params != self.params:
        #             print("New params differ, using former params instead...")
        #             params = self.params

    def train(self):
        total_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_parameters}")
        output_dir = self.output_dir
        if os.path.exists(f"{output_dir}/chkpts/best.pt"):
            print("Checkpoint already exists.")
            # if input("Resume training? (y/n)") != "y":
            #     return
        else:
            output_dir = f"{output_dir}/{datetime.now().strftime('%m-%d_%H%M%S')}"
            print(f"Creating new log folder as {output_dir}")
        learner = StrucMusDiffLearner(
            output_dir, self.model_name, self.model, self.train_dl, self.val_dl, self.optimizer,
            self.params
        )
        learner.train(max_epoch=self.params.max_epoch,is_inpaint=self.is_inpaint)
