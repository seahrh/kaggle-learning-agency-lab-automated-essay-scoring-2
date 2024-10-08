import gc
import json
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scml
import torch
from pytorch_lightning.loggers import CSVLogger
from scml import pandasx as pdx
from scml import torchx
from sklearn.metrics import f1_score, precision_score, recall_score

# from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (  # DebertaV2Model,; DebertaV2PreTrainedModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

import lalaes2 as mylib

# from transformers.modeling_outputs import TokenClassifierOutput

__all__ = [
    "PersuadeDataset",
    "PersuadeModel",
    "PersuadeTask",
    "classify_persuade_topic_logits",
    "classify_persuade_topic",
    "evaluation_persuade",
]

log = scml.get_logger(__name__)


class PersuadeDataset(Dataset):

    ID_TO_LABEL: Dict[int, str] = {
        0: "facial action coding system",
        1: "distance learning",
        2: "electoral college work",
        3: "car free cities",
        4: "driverless cars",
        5: "exploring venus",
        6: "summer projects",
        7: "mandatory extracurricular activities",
        8: "cell phones school",
        9: "grades extracurricular activities",
        10: "face mars",
        11: "seeking multiple opinions",
        12: "community service",
        13: "cowboy rode waves",
        14: "phones driving",
    }
    LABEL_TO_ID: Dict[str, int] = {v: k for k, v in ID_TO_LABEL.items()}
    EVALUATION_CLASSES = sorted(list(ID_TO_LABEL.keys()))

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        texts: List[str],
        labels: Optional[List[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = [] if labels is None else labels

    def __getitem__(self, idx):
        res = {}
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            is_split_into_words=False,  # processing an array of string tokens (instead of a string)
            add_special_tokens=True,
            return_overflowing_tokens=False,  # no striding (overlapping tokens)
            return_offsets_mapping=False,
            return_special_tokens_mask=False,
        )
        for k, v in enc.items():
            t = torch.tensor(v)
            log.debug(f"{k} {t.shape}")
            res[k] = t
        if len(self.labels) != 0:
            res["labels"] = torch.tensor(
                self.labels[idx],
                # use int64 instead of int32 to prevent error on nvidia gpu
                # https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
                dtype=torch.int64,
            )
        return res

    def __len__(self):
        return len(self.texts)


def classify_persuade_topic_logits(
    ds: PersuadeDataset,
    model: PreTrainedModel,
    batch_size: int,
    device: Optional[torch.device] = None,
    progress_bar: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")
    torch.cuda.empty_cache()
    res = []
    batches = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()  # type: ignore
    model.to(device)  # type: ignore
    with torch.no_grad():
        for batch in tqdm(
            batches,
            desc="persuade topic classification",
            disable=not progress_bar,
            mininterval=10,
        ):
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)  # type: ignore
            # (batch_size, sequence_length, config.num_labels)
            logits = outputs.logits.detach().cpu()
            log.debug(f"logits.size={logits.size()}\n{logits}")
            res += logits.tolist()
    return np.array(res, dtype=dtype)


def classify_persuade_topic(
    ds: PersuadeDataset,
    model: PreTrainedModel,
    batch_size: int,
    device: Optional[torch.device] = None,
    progress_bar: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    logits = classify_persuade_topic_logits(
        ds=ds,
        model=model,
        batch_size=batch_size,
        device=device,
        progress_bar=progress_bar,
        dtype=dtype,
    )
    y_proba = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    return np.array(y_proba, dtype=dtype)


def evaluation_persuade(
    ds: PersuadeDataset,
    model: PreTrainedModel,
    batch_size: int,
    device: Optional[torch.device] = None,
    progress_bar: bool = True,
) -> Dict:
    y_true: List[int] = [int(ds[i]["labels"].item()) for i in range(len(ds))]
    y_pred = np.argmax(
        classify_persuade_topic_logits(
            ds=ds,
            model=model,
            batch_size=batch_size,
            device=device,
            dtype=np.float32,
            progress_bar=progress_bar,
        ),
        axis=1,
    ).flatten()
    return {
        "f1": f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            labels=PersuadeDataset.EVALUATION_CLASSES,
        ),
        "recall": recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            labels=PersuadeDataset.EVALUATION_CLASSES,
        ),
        "precision": precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
            labels=PersuadeDataset.EVALUATION_CLASSES,
        ),
    }


# noinspection PyAbstractClass
class PersuadeModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_dir: str,
        lr: float,
        scheduler_conf: Iterable[SectionProxy],
        model_class: str = "auto",
        swa_start_epoch: int = -1,
        gradient_checkpointing: bool = False,
        hidden_dropout_prob: Optional[float] = None,
        attention_probs_dropout_prob: Optional[float] = None,
        max_position_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.hparams.scheduler_conf = list(scheduler_conf)
        for s in self.hparams.scheduler_conf:
            if s["qualified_name"] == "torch.optim.swa_utils.SWALR":
                s["swa_lr"] = str(self.hparams.lr)
        self.model: PreTrainedModel = self.pretrained_model()
        self.swa_enable: bool = self.hparams.swa_start_epoch >= 0
        # only `AveragedModel.module` is used in its forward pass, so we save `AveragedModel.module`
        # see https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
        if self.swa_enable:
            self.automatic_optimization = False
            self.swa_model = torch.optim.swa_utils.AveragedModel(model=self.model)
            self.model = self.swa_model.module
        self._has_swa_started: bool = False

    def pretrained_model(self) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(self.hparams.pretrained_dir)
        config.problem_type = "single_label_classification"
        config.id2label = PersuadeDataset.ID_TO_LABEL
        config.label2id = PersuadeDataset.LABEL_TO_ID
        setattr(config, "gradient_checkpointing", self.hparams.gradient_checkpointing)
        if self.hparams.hidden_dropout_prob is not None:
            setattr(config, "hidden_dropout_prob", self.hparams.hidden_dropout_prob)
        if self.hparams.attention_probs_dropout_prob is not None:
            setattr(
                config,
                "attention_probs_dropout_prob",
                self.hparams.attention_probs_dropout_prob,
            )
        # max_position_embeddings for deberta-v2
        if self.hparams.max_position_embeddings is not None:
            setattr(
                config,
                "max_position_embeddings",
                self.hparams.max_position_embeddings,
            )
        log.info(f"config.to_diff_dict={json.dumps(config.to_diff_dict(), indent=2)}")
        if self.hparams.model_class == "CustomDebertaV2ForTokenClassification":
            raise NotImplementedError("custom class not implemented yet!")
        return AutoModelForSequenceClassification.from_pretrained(
            self.hparams.pretrained_dir,
            config=config,
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if not self.automatic_optimization:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
            for opt in opts:
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
        return loss

    def on_train_epoch_end(self):
        if self.automatic_optimization:
            return
        epoch = self.trainer.current_epoch
        schedulers = self.lr_schedulers()
        if self.swa_enable:
            if epoch >= self.hparams.swa_start_epoch:
                self._has_swa_started = True
                self.swa_model.update_parameters(self.model)
                for sch in schedulers:
                    if isinstance(sch, torch.optim.swa_utils.SWALR):
                        sch.step()
                return
            schedulers = [
                sch
                for sch in schedulers
                if not isinstance(sch, torch.optim.swa_utils.SWALR)
            ]
        if schedulers is None:
            return
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for sch in schedulers:
            if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step()

    def _shared_eval_step(self, batch, batch_idx):
        model = self.model
        if self._has_swa_started:
            model = self.swa_model
        outputs = model(**batch)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if not self.automatic_optimization:
            for sch in self.lr_schedulers():
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(loss)

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        """

        :return: Two lists - The first list has multiple optimizers,
        and the second has multiple LR schedulers (or multiple lr_scheduler_config).
        """
        # The optimizer does not seem to reference any FSDP parameters.
        # HINT: Make sure to create the optimizer after setting up the model
        # by referencing `self.trainer.model.parameters()` in the `configure_optimizers()` hook.
        optimizers = [
            torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.hparams.lr,
                amsgrad=False,
            )
        ]
        schedulers = torchx.schedulers_by_config(
            optimizer=optimizers[0], sections=self.hparams.scheduler_conf
        )
        if self.hparams.swa_start_epoch >= 0:
            if len(schedulers) == 0:
                raise ValueError("For SWA, there must be at least one scheduler")
            if not isinstance(schedulers[0].scheduler, torch.optim.swa_utils.SWALR):
                raise ValueError(
                    "For SWA, the first scheduler must be of the type `torch.optim.swa_utils.SWALR`"
                )
        return optimizers, [s._asdict() for s in schedulers]


class PersuadeTask(mylib.Task):
    name: str = "persuade"

    def __init__(
        self,
        conf: ConfigParser,
    ):
        super().__init__(conf=conf)
        self.tra_ds: Optional[PersuadeDataset] = None
        self.val_ds: Optional[PersuadeDataset] = None
        self.trainer: Optional[pl.Trainer] = None
        self.batch_size: int = self.conf.getint("batch_size")
        self.eval_every_n_steps: int = self.conf.getint("eval_every_n_steps")
        self.callbacks = mylib.training_callbacks(
            patience=self.conf.getint("patience"),
            eval_every_n_steps=self.eval_every_n_steps,
            ckpt_filename=self.conf.get("ckpt_filename", ""),
            save_top_k=self.conf.getint("ckpt_save_top_k"),
        )

    def _best_model(self) -> PersuadeModel:
        if self.trainer is None:
            raise ValueError("Trainer must not be null")
        ckpt_path: str = self.trainer.checkpoint_callback.best_model_path
        log.info(f"best_model_path={ckpt_path}")
        return PersuadeModel.load_from_checkpoint(ckpt_path)  # type: ignore[no-any-return]

    def _get_datasets(self) -> None:
        log.info("Prepare dataset...")
        with scml.Timer() as tim:
            tokenizer = AutoTokenizer.from_pretrained(
                self.mc["directory"],
                model_max_length=self.conf.getint("model_max_length"),
            )
            train_data_first_n: int = self.conf.getint("train_data_first_n")
            filepath = self.conf["train_data_file"]
            df = pd.read_parquet(filepath)
            if 0 < train_data_first_n < len(df):
                df = df.iloc[:train_data_first_n]
            log.info(f"filepath={filepath}\n{pdx.info_string(df)}")
            self.tra_ds = PersuadeDataset(
                tokenizer=tokenizer,
                texts=df["full_text"].tolist(),
                labels=[PersuadeDataset.LABEL_TO_ID[topic] for topic in df["topic"]],
            )
            filepath = self.conf["validation_data_file"]
            df = pd.read_parquet(filepath)
            log.info(f"filepath={filepath}\n{pdx.info_string(df)}")
            self.val_ds = PersuadeDataset(
                tokenizer=tokenizer,
                texts=df["full_text"].tolist(),
                labels=[PersuadeDataset.LABEL_TO_ID[topic] for topic in df["topic"]],
            )
            del df
            gc.collect()
        log.info(f"Prepare dataset...DONE. Time taken {str(tim.elapsed)}")

    def _evaluation(
        self, model: PreTrainedModel, epochs: int, device: Optional[torch.device] = None
    ) -> None:
        if self.val_ds is None:
            raise ValueError("validation dataset must not be None")
        log.info("Evaluation...")
        with scml.Timer() as tim:
            self.validation_result = {
                "train_epochs": epochs,
            }
            self.validation_result.update(
                evaluation_persuade(
                    ds=self.val_ds,
                    model=model,
                    batch_size=self.batch_size * 8,
                    device=device,
                )
            )
        log.info(f"Evaluation...DONE. Time taken {str(tim.elapsed)}")

    def _train_final_model(
        self,
        hps: Dict[str, mylib.ParamType],
    ) -> None:
        if self.tra_ds is None:
            raise ValueError("train dataset must not be None")
        if self.val_ds is None:
            raise ValueError("validation dataset must not be None")
        log.info("Train final model on best Hps...")
        log.info(f"hps={hps}")
        gc.collect()
        torch.cuda.empty_cache()
        with scml.Timer() as tim:
            log.info(f"len(tra)={len(self.tra_ds):,}, len(val)={len(self.val_ds):,}")
            self.trainer = pl.Trainer(
                default_root_dir=self.conf["job_dir"],
                strategy=self.conf.get("train_strategy", "auto"),
                precision=self.conf.get("train_precision", None),
                accelerator=self.accelerator,
                devices=self.devices,
                max_epochs=self.conf.getint("epochs"),
                check_val_every_n_epoch=None if self.eval_every_n_steps > 0 else 1,
                val_check_interval=(
                    self.eval_every_n_steps if self.eval_every_n_steps > 0 else 1.0
                ),
                callbacks=self.callbacks,
                deterministic=False,
                logger=CSVLogger(save_dir=self.conf["job_dir"]),
            )
            log.info(f"trainer.precision={self.trainer.precision}")
            log.info(f"model_class={self.conf.get('model_class', 'auto')}")
            num_workers: int = self.conf.getint("dataloader_num_workers")
            ckpt_path: Optional[str] = self.conf.get("resume_training_from", "")
            if ckpt_path is not None and len(ckpt_path) == 0:
                ckpt_path = None
            self.trainer.fit(
                model=PersuadeModel(
                    pretrained_dir=self.mc["directory"],
                    lr=float(hps["lr"]),
                    swa_start_epoch=int(hps["swa_start_epoch"]),
                    scheduler_conf=self.scheduler_conf,
                    model_class=self.conf.get("model_class", "auto"),
                    max_position_embeddings=(
                        self.conf.getint("model_max_length")
                        if "model_max_length" in self.conf
                        else None
                    ),
                    gradient_checkpointing=(
                        self.conf.getboolean("gradient_checkpointing")
                        if "gradient_checkpointing" in self.conf
                        else False
                    ),
                    hidden_dropout_prob=(
                        self.conf.getfloat("hidden_dropout_prob")
                        if "hidden_dropout_prob" in self.conf
                        else None
                    ),
                    attention_probs_dropout_prob=(
                        self.conf.getfloat("attention_probs_dropout_prob")
                        if "attention_probs_dropout_prob" in self.conf
                        else None
                    ),
                ),
                train_dataloaders=DataLoader(
                    self.tra_ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    persistent_workers=True if num_workers > 0 else False,
                ),
                val_dataloaders=DataLoader(
                    self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    persistent_workers=True if num_workers > 0 else False,
                ),
                ckpt_path=ckpt_path,
            )
        log.info(f"Train final model on best Hps...DONE. Time taken {str(tim.elapsed)}")

    def run(self) -> None:
        with scml.Timer() as tim:
            self._get_datasets()
            self._train_final_model(
                hps={
                    "lr": self.conf.getfloat("lr"),
                    "swa_start_epoch": self.conf.getfloat("swa_start_epoch"),
                },
            )
            torch.distributed.barrier()
            if self.trainer is not None:
                if self.trainer.received_sigterm:
                    log.info("Exit now because signal.SIGTERM signal was received.")
                    return
                if self.trainer.is_global_zero:
                    best: PersuadeModel = self._best_model()
                    self.save_hf_model(
                        model=best.model,
                        dst_path=Path(self.conf["job_dir"]),
                    )
                    self._evaluation(
                        model=best.model,
                        epochs=self.trainer.current_epoch,
                        device=self._device(),
                    )
                    self._save_job_config()
                    self._copy_tokenizer_files(src=Path(self.mc["directory"]))
        if self.trainer is not None and self.trainer.is_global_zero:
            log.info(
                f"Total time taken {str(tim.elapsed)}. Saved {self.conf['job_dir']}"
            )
