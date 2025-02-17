from argparse import ArgumentParser
from params.params_ddpm_pp import params as params_pp
from params.params_ddpm_str import params as params_str
from train.train_ldm import LDM_TrainConfig

if __name__ == "__main__":
    parser = ArgumentParser(
        description='train (or resume training) a StrucMusDiff model'
    )

    parser.add_argument(
        "--output_dir",
        default='saved_models/',
        help='directory in which to store model checkpoints and training logs'
    )
    parser.add_argument("--model", default="str", help="which model to train")
    args = parser.parse_args()

    if args.model.find("pp")!=-1:
        config = LDM_TrainConfig(params_pp, args.output_dir)
    elif args.model.find("str")!=-1:
        config = LDM_TrainConfig(params_str, args.output_dir, args.model, is_inpaint=True)
    config.train()
