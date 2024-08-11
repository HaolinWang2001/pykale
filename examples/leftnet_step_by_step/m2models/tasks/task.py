"""
Codes borrowed from Open Catalyst Project (OCP) https://github.com/Open-Catalyst-Project/ocp (MIT license)
"""

import logging
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from m2models.common.registry import registry
from torch_geometric.data import DataLoader

from examples.leftnet_step_by_step.get_model import load_model
from examples.leftnet_step_by_step.m2models.common.data_parallel import ParallelCollater
from examples.leftnet_step_by_step.m2models.datasets.lmdb_dataaccess import LmdbDatasetAccess


# from m2models.trainers.forces_trainer import ForcesTrainer


class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        if self.config["checkpoint"] is not None:
            self.trainer.load_checkpoint(self.config["checkpoint"])

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

    def run(self):
        # try:
        #     self.trainer.train(
        #         disable_eval_tqdm=self.config.get(
        #             "hide_eval_progressbar", False
        #         )
        #     )
        # except RuntimeError as e:
        #     self._process_error(e)
        #     raise e
        parallel_collater = ParallelCollater(
                    0 if self.config["cpu"] else 1,
            #         # self.config["model_attributes"].get("otf_graph", False),
                )
        dataset = LmdbDatasetAccess(self.config["dataset"]["train"], self.config["dataset"]["val"], self.config["dataset"]["test"])
        train_dataset = dataset.get_train()
        val_dataset = dataset.get_valid()
        test_dataset = dataset.get_test()
        train_loader = DataLoader(train_dataset, collate_fn=parallel_collater, batch_size=self.config["optim"]["batch_size"],
               shuffle=True, pin_memory=True, )
        val_loader = DataLoader(val_dataset, collate_fn=parallel_collater, batch_size=self.config["optim"]["batch_size"], shuffle=False, pin_memory=True, )
        test_loader = DataLoader(test_dataset, collate_fn=parallel_collater, batch_size=self.config["optim"]["batch_size"], shuffle=False, pin_memory=True, )

        model = load_model(self.config, dataset, 1)
        # model = self.trainer
        trainer = pl.Trainer(
            max_epochs=self.config["optim"]["max_epochs"],
            accelerator="gpu" if not self.config["cpu"] else "cpu",
            devices="auto",
            logger=pl_loggers.TensorBoardLogger("./"),
            # registry.get_logger_class(self.config["logger"])(self.config),


        )
        trainer.fit(model, train_loader, val_loader)





@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self):
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        results_file = "predictions"
        self.trainer.predict(
            self.trainer.test_loader,
            results_file=results_file,
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self):
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("run-relaxations")
class RelxationTask(BaseTask):
    def run(self):
        assert isinstance(
            self.trainer, ForcesTrainer
        ), "Relaxations are only possible for ForcesTrainer"
        assert (
            self.trainer.relax_dataset is not None
        ), "Relax dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.run_relaxations()
