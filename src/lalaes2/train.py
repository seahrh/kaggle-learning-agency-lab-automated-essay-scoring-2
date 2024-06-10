import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import scml
import torch
from scml import configparserx as cpx
from tqdm import tqdm

import lalaes2 as mylib

tqdm.pandas()


log = scml.get_logger(__name__)


def _set_env() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    pd.set_option("max_info_columns", 9999)
    pd.set_option("display.max_columns", 9999)
    pd.set_option("display.max_rows", 9999)
    pd.set_option("max_colwidth", 9999)


def _check_device() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log.info(
                f"device={i}, {torch.cuda.get_device_name(i)}"
                f"\nMem Allocated: {round(torch.cuda.memory_allocated(i) / 1024 ** 3, 1)} GB"
                f"\nMem Cached: {round(torch.cuda.memory_reserved(i) / 1024**3, 1)} GB"
            )
    else:
        log.info("cuda not available")


def _main(argv=None):
    _set_env()
    _check_device()
    node_rank = int(os.environ.get("NODE_RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    log.info(f"NODE_RANK={node_rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--home-dir",
        dest="home_dir",
        default="",
        required=False,
        help="path to home directory",
    )
    parser.add_argument(
        "--task",
        dest="task_name",
        required=True,
        help="Task type e.g. 'ner'",
    )
    parser.add_argument(
        "--conf", dest="conf", required=True, help="filepath to job.ini"
    )
    args, unknown_args = parser.parse_known_args(argv)
    conf = cpx.load(Path(args.conf))
    pl.seed_everything(conf["DEFAULT"].getint("seed"))
    if args.home_dir != "":
        conf["DEFAULT"]["home_dir"] = str(Path(args.home_dir).resolve())
    model_dir: str = conf[args.task_name]["model_dir"]
    backbone: str = ""
    if "backbone" in conf[args.task_name]:
        backbone = conf[args.task_name]["backbone"]
    if backbone == "" and "teacher_backbone" in conf[args.task_name]:
        backbone = conf[args.task_name]["teacher_backbone"]
    if backbone == "":
        raise ValueError("backbone must not be empty string")
    conf[args.task_name]["backbone"] = backbone
    log.info(f"args={args}\nunknown_args={unknown_args}")
    log.info(cpx.as_dict(conf))
    task = None
    if args.task_name == mylib.Aes2Task.name:
        task = mylib.Aes2Task(
            conf=conf,
        )
    if task is None:
        raise NotImplementedError(f"Unsupported task type: {args.task_name}")
    s: str = conf[args.task_name].get("job_dir", "")
    if len(s) == 0:
        job_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = Path(model_dir) / args.task_name / backbone / job_ts
        job_dir.mkdir(parents=True, exist_ok=True)
        conf["DEFAULT"]["job_dir"] = str(job_dir.resolve())
    elif not Path(s).is_dir():
        raise ValueError(f"job_dir does not exist [{s}]")
    task.run()


if __name__ == "__main__":
    _main()
