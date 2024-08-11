import datetime
import logging
import os

import torch
import torch_geometric
from torch import nn
from torch_geometric.data import DataLoader
from tqdm import tqdm

from m2models.common import distutils
from m2models.common.registry import registry
from m2models.modules.scaling.util import ensure_fitted
from m2models.trainers.base_trainer import BaseTrainer
from m2models.modules.evaluator import Evaluator


from m2models.trainers.base_trainer import BaseTrainer

from examples.leftnet_step_by_step.m2models.common.data_parallel import ParallelCollater
from examples.leftnet_step_by_step.m2models.datasets.lmdb_dataaccess import LmdbDatasetAccess
from kale.pipeline.base_nn_trainer import BaseNNTrainer

@registry.register_trainer("pykaleenergy")
class PykaleEnergyTrainer(BaseNNTrainer):
#
#     def __init__(
#             self,
#             task,
#             model,
#             dataset,
#             optimizer,
#             identifier,
#             normalizer=None,
#             timestamp_id=None,
#             run_dir=None,
#             is_debug=False,
#             is_hpo=False,
#             print_every=100,
#             seed=None,
#             logger="tensorboard",
#             local_rank=0,
#             amp=False,
#             cpu=False,
#             slurm={},
#             noddp=False,
#         ):
#         super().__init__(
#             task=task,
#             model=model,
#             dataset=dataset,
#             optimizer=optimizer,
#             identifier=identifier,
#             normalizer=normalizer,
#             timestamp_id=timestamp_id,
#             run_dir=run_dir,
#             is_debug=is_debug,
#             is_hpo=is_hpo,
#             print_every=print_every,
#             seed=seed,
#             logger=logger,
#             local_rank=local_rank,
#             amp=amp,
#             cpu=cpu,
#             name="is2re",
#             slurm=slurm,
#             noddp=noddp,
#         )

    def __init__(self, model, max_epochs, init_lr=0.001, optimizer_params=None, adapt_lr=False, run_dir=None,
                 seed=None, logger=None):
        super().__init__(optimizer_params, max_epochs, init_lr, adapt_lr)
        self.model = model
        timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
            self.device
        )
        # create directories from master rank only
        distutils.broadcast(timestamp, 0)
        timestamp = datetime.datetime.fromtimestamp(
            timestamp.int()
        ).strftime("%Y-%m-%d-%H-%M-%S")

        self.timestamp_id = timestamp

        self.config = {
            "trainer": "pykaleenergy",
            "model": "leftnet",
            # "dataset": dataset,
            "model_attributes": model,
            "optim": optimizer_params,
            "logger": logger,
            # "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                # "identifier": identifier,
                # "print_every": print_every,
                # "seed": seed,
                # "timestamp_id": self.timestamp_id,
                # "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", self.timestamp_id
                ),
                "results_dir": os.path.join(
                    run_dir, "results", self.timestamp_id
                ),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger, self.timestamp_id
                ),
            },
            # "slurm": slurm,
            # "noddp": noddp,
        }
        # self.load_datasets()

    def load_task(self):
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.num_targets = 1

    def forward(self, batch_list):
        output = self.model(batch_list[0])
        return output

    def compute_loss(self, batch_list, split_name="valid"):
        output = self.forward(batch_list)
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )

        loss = nn.L1Loss()(output, energy_target)
        # return loss

        log_metrics = {f"{split_name}_loss": loss}
        return loss, log_metrics

    # def _compute_loss

    def configure_optimizers(self):
        # super().load_optimizer()
        # optimizer = self.config["optim"].get("optimizer", "AdamW")
        # optimizer = getattr(optim, optimizer)
        # self.optimizer = optimizer(
        #     params=self.model.parameters(),
        #     lr=self.config["optim"]["lr_initial"],
        #     **self.config["optim"].get("optimizer_params", {}),
        # )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["optim"]["lr_initial"],
            **self.config["optim"].get("optimizer_params", {}),
        )

        return [optimizer]

