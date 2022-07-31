import numpy as np
import os
import tqdm
import torch
import random
import logging

from .dataset import DataLoader
from .opt import OpenAIAdam
from .prediction_model import CLMBR

from .. import timeline
from .. import ontology
from .. import labeler

from ..utils import set_up_logging
from ..extension.clmbr import PatientTimelineDataset

from typing import Optional


class Trainer:
    def __init__(self, model: CLMBR, log_path: Optional[str] = None):
        if log_path is not None:
            set_up_logging(log_path)
        self.model = model

    def _build_adam_optimizer(
        self, dataset: PatientTimelineDataset
    ) -> OpenAIAdam:
        config = self.model.config
        params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            params.append(param)
        optimizer = OpenAIAdam(
            params,
            lr=config["lr"],
            schedule="warmup_linear",
            warmup=config["warmup_epochs"] / config["epochs_per_cycle"],
            t_total=dataset.num_batches(config["batch_size"], False)
            * config["epochs_per_cycle"],
            b1=config["b1"],
            b2=config["b2"],
            e=config["e"],
            l2=config["l2"],
        )
        logging.info(
            f"Batches per epoch = {dataset.num_batches(config['batch_size'], False)}"
        )
        logging.info(f"Total batches = {optimizer.defaults['t_total']}")
        return optimizer

    def _train_epoch(
        self, dataset: PatientTimelineDataset, pbar: Optional[tqdm.tqdm] = None
    ) -> None:
        self.model.train()
        total_non_text_loss = 0
        config = self.model.config
        with DataLoader(
            dataset,
            is_val=False,
            batch_size=config["batch_size"],
            seed=random.randint(0, 100000),
            day_dropout=config["day_dropout"],
            code_dropout=config["code_dropout"],
        ) as batches:
            for i, batch in enumerate(batches):
                # train with mixed precision
                if self.scaler is not None:
                    # casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)

                    # scales the loss, and calls backward() to create scaled gradients
                    self.scaler.scale(outputs["loss"]).backward()

                    # unscales gradients and calls or skips optimizer.step()
                    self.scaler.step(self.optimizer)

                    # updates the scale for next iteration
                    self.scaler.update()

                else:
                    outputs = self.model(batch)

                    self.optimizer.zero_grad()
                    outputs["loss"].backward()
                    self.optimizer.step()

                del outputs
                del batch
                if pbar is not None:
                    pbar.update(1)
                elif i % 2000 == 0:
                    logging.info(f"Seen batch {i}")

    def train(
        self, dataset: PatientTimelineDataset, use_pbar: bool = True
    ) -> None:
        self.model.train()

        model_dir = self.model.config["model_dir"]
        num_epochs = self.model.config["epochs_per_cycle"]

        self.optimizer = self._build_adam_optimizer(dataset)

        if self.model.config["mixed_precision_training"]:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_loss = None
        best_val_loss_epoch = None
        last_val_loss = np.inf
        trigger_early_stopping = 0

        pbar = (
            tqdm.tqdm(total=self.optimizer.defaults["t_total"])
            if use_pbar
            else None
        )
        loss_file = open(os.path.join(model_dir, "losses"), "w")

        logging.info("Start training")
        for epoch in range(num_epochs):
            logging.info("About to start epoch %s", epoch)
            if pbar is not None:
                pbar.set_description(f"Epoch {epoch}")
            self._train_epoch(dataset, pbar=pbar)
            logging.info("Epoch %s is complete", epoch)

            if pbar is not None:
                pbar.set_description(f"Evaluating epoch {epoch}")
            train_loss = self.evaluate(dataset, is_val=False, num_batches=2000)
            val_loss = self.evaluate(dataset, is_val=True, num_batches=2000)

            logging.info("Train loss: %s", train_loss)
            logging.info("Val loss: %s", val_loss)

            loss_file.write("Epoch {}\n".format(epoch))
            loss_file.write("Train loss {}\n".format(train_loss))
            loss_file.write("Val loss {}\n".format(val_loss))
            loss_file.write("\n")
            loss_file.flush()

            # checkpoint best model
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch

                best_path = os.path.join(model_dir, "best")
                if os.path.exists(best_path):
                    os.unlink(best_path)

                torch.save(self.model.state_dict(), best_path)
                logging.info("Saving best model to %s", best_path)

            # early stopping
            if self.model.config["early_stopping"] and val_loss > last_val_loss:
                trigger_early_stopping += 1

                if trigger_early_stopping >= self.model.config["early_stopping_patience"]:
                    logging.info("Stopping early at epoch %s!", epoch)
                    if pbar is not None:
                        pbar.close()
                    loss_file.close()
                    logging.info("Training complete!")
                    break
            else:
                trigger_early_stopping = 0

            last_val_loss = val_loss

        if pbar is not None:
            pbar.close()
        loss_file.close()
        logging.info("Training complete!")

    def evaluate(
        self,
        dataset: PatientTimelineDataset,
        is_val: bool = True,
        num_batches: Optional[int] = None,
    ) -> float:
        self.model.eval()
        config = self.model.config
        if num_batches is None:
            num_batches = dataset.num_batches(config["batch_size"], is_val)
        total_loss = 0
        with DataLoader(
            dataset,
            is_val=is_val,
            batch_size=config["eval_batch_size"],
            seed=0,
            day_dropout=0,
            code_dropout=0,
        ) as batches:
            for batch, _ in zip(batches, range(num_batches)):
                with torch.no_grad():
                    outputs = self.model(batch)

                    total_loss += outputs["loss"].item()

                    del batch
                    del outputs

        return total_loss / num_batches
