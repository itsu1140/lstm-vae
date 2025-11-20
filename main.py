import argparse
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml
from src.config.params import Params
from src.models.pred import pred
from src.plot import plot
from src.train.train import Train
from src.utils.logger import setup_logger


def train() -> None:
    # prepare directories
    jst: timezone = timezone(timedelta(hours=+9), "JST")
    date: str = datetime.now(jst).strftime("%Y%m%d")
    outputs: Path = Path("outputs/")
    date_dir_count: int = (
        sum(1 for p in (outputs / "logs").iterdir() if p.is_dir() and date in p.name)
        + 1
    )
    out_path: Path = Path(f"{date}-{date_dir_count}")
    log_path: Path = outputs / "logs" / out_path
    log_path.mkdir()
    model_path: Path = outputs / "models" / out_path
    model_path.mkdir()

    logger = setup_logger(log_path)

    # training
    piano_size = 88
    params: Params = Params(
        # model
        piano_size=piano_size,
        input_size=piano_size + 12,
        output_size=12,
        seq_len=32,
        batch_size=128,
        hidden_size=128,
        latent_size=64,
        num_layers=2,
        beta_max=1,
        # train
        epoch=300,
        # midi
        offset=24,
        unittime=0.25,
    )

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = Train(params=params)
    start: str = datetime.now(jst).strftime("%Y%m%d %H:%M:%S")
    logger.info(f"Device: {device.type}\nStart training at {start}")
    tr.train()
    logger.info("Training finished")

    # save model and params
    torch.save(
        tr.model.state_dict(),
        model_path / "model.pt",
    )
    with Path.open(model_path / "params.yaml", "w") as f:
        yaml.safe_dump(asdict(params), f)
    plot.tsne(label2genre=tr.label2genre, history=tr.history, output_path=out_path)


def chk_path(model_path: Path, data_path: Path, args: argparse.Namespace) -> Path:
    model_path_parts: list[str] = list(model_path.parts)
    model_path_parts.reverse()
    model_parent: str = ""
    for part in model_path_parts:
        if re.fullmatch(r"\d{8}-\d+", part):
            model_parent = part
            break
    model_dir: Path = Path("outputs/models") / model_parent
    if not ((model_dir / "model.pt").exists() and data_path.exists()):
        error_msg = f"{args.model} or {args.input} is not exist"
        raise FileNotFoundError(error_msg)
    return model_dir


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "pred"], required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--input", type=str)
    args: argparse.Namespace = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "pred":
        model_path: Path = Path(args.model)
        data_path: Path = Path(args.input)
        model_dir: Path = chk_path(model_path, data_path, args)
        pred(
            model_dir=model_dir,
            data_path=data_path,
        )
    else:
        print("mode should be 'train' or 'pred'")


if __name__ == "__main__":
    main()
