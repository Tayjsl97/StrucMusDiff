from . import AttrDict

params = AttrDict(
    data_path='data_process/All_train_data.data',
    # Training params
    batch_size=16,
    max_epoch=200,
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

    # Data params
    num_workers=0,
    pin_memory=False,

    # unet
    in_channels=1,
    out_channels=1,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=128,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,

    # img
    img_h=128,
    img_w=128,

    # conditional
    cond_type="pp",
    cond_mode="cond",  # {cond, uncond}

    # Similar_Phrase_Encoder
    use_enc=True,
    SPE_layers=8,
    SPE_heads=8,
    width=128,
    dropout=0.1,
    activate='gelu',
    attention_type='linear'
)
