from . import *
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.latent_diffusion import LatentDiffusion
from models.StrucMusDiff import StrucMusDiff
from models.Reference_Phrase_Encoder import Reference_Phrase_Encoder
from .utils import get_train_val_dataloaders

class LDM_TrainConfig(TrainConfig):
    def __init__(
        self,
        params,
        output_dir,
        is_inpaint=False,
    ) -> None:
        super().__init__(params, None, output_dir, is_inpaint)
        self.autoencoder = None

        self.unet_model = UNetModel(
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

        self.ldm_model = LatentDiffusion(
            linear_start=params.linear_start,
            linear_end=params.linear_end,
            n_steps=params.n_steps,
            # latent_scaling_factor=params.latent_scaling_factor,
            autoencoder=self.autoencoder,
            unet_model=self.unet_model
        )
        if params.use_enc:
            self.similar_phrase_enc= Reference_Phrase_Encoder(params.SPE_layers,params.SPE_heads,params.width,
                                                                 params.dropout,params.activate,params.attention_type)
        else:
            self.similar_phrase_enc = None
        self.model = StrucMusDiff(
            self.ldm_model,
            cond_type=params.cond_type,
            cond_mode=params.cond_mode,
            similar_phrase_enc=self.similar_phrase_enc
        ).to(self.device)
        # Create dataloader
        self.train_dl, self.val_dl = get_train_val_dataloaders(
                params.data_path,
                params.batch_size,
                params.num_workers,
                params.pin_memory
            )

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
