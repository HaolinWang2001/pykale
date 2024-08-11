import argparse
import logging

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from examples.leftnet_step_by_step.m2models.common.data_parallel import ParallelCollater
from examples.leftnet_step_by_step.m2models.datasets.lmdb_dataaccess import LmdbDatasetAccess
# from torch_geometric.data import DataLoader

# from config import get_cfg_defaults
# from model import get_model
from get_model import load_model
# from data import get_data
# from data import Data
# from data import *
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

import yaml

# embeddings
# from qmof_khot_embeddings import QMOF_KHOT_EMBEDDINGS
# from khot_embeddings import KHOT_EMBEDDINGS


# from kale.loaddata.video_access import VideoDataset
# from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets
from kale.utils.seed import set_seed
# from lmdb_data import LmdbDataset
# import numpy as np
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Domain Adversarial Networks on Action Datasets")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help="gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    """The main for this domain adaptation example, showing the workflow"""
    args = arg_parse()

    # ---- setup configs ----
    # cfg = get_cfg_defaults()
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    # cfg.merge_from_file(args.cfg)
    # cfg.freeze()
    print(cfg)

    # ---- setup output ----
    format_str = "@%(asctime)s %(name)s [%(levelname)s] - (%(message)s)"
    logging.basicConfig(format=format_str)
    # ---- setup dataset ----
    # seed = cfg["model"]["seed"] if cfg["model"]["seed"] else 2020
    seed = 2020
    # source, target, num_classes = VideoDataset.get_source_target(
    #     VideoDataset(cfg.DATASET.SOURCE.upper()), VideoDataset(cfg.DATASET.TARGET.upper()), seed, cfg
    # )
    # dataset = VideoMultiDomainDatasets(
    #     source,
    #     target,
    #     image_modality=cfg.DATASET.IMAGE_MODALITY,
    #     seed=seed,
    #     config_weight_type=cfg.DATASET.WEIGHT_TYPE,
    #     config_size_type=cfg.DATASET.SIZE_TYPE,
    # )
    parallel_collater = ParallelCollater(
        # 0 if cfg["cpu"] else 1,
        1,
        #         # self.config["model_attributes"].get("otf_graph", False),
    )
    dataset = LmdbDatasetAccess(cfg["dataset"]["train"], cfg["dataset"]["val"],
                                cfg["dataset"]["test"])
    train_dataset = dataset.get_train()
    val_dataset = dataset.get_valid()
    test_dataset = dataset.get_test()
    train_loader = DataLoader(train_dataset, collate_fn=parallel_collater,
                              batch_size=cfg["optim"]["batch_size"],
                              shuffle=True, pin_memory=True, )
    val_loader = DataLoader(val_dataset, collate_fn=parallel_collater, batch_size=cfg["optim"]["batch_size"],
                            shuffle=False, pin_memory=True, )
    test_loader = DataLoader(test_dataset, collate_fn=parallel_collater, batch_size=cfg["optim"]["batch_size"],
                             shuffle=False, pin_memory=True, )



    # ---- training/test process ----
    ### Repeat multiple times to get std
    # for i in range(0, cfg.DATASET.REPEAT):
    for i in range(0, 1):
        seed = seed + i * 10
        set_seed(seed)  # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        print(f"==> Building model for seed {seed} ......")
        # ---- setup model and logger ----
        model = load_model(cfg, dataset, cfg["model"]["output_dim"])
        tb_logger = pl_loggers.TensorBoardLogger('./', name="seed{}".format(seed))
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            # save_last=True,
            # save_top_k=1,
            monitor="valid_loss",
            mode="min",
        )

        ### Set early stopping
        # early_stop_callback = EarlyStopping(monitor="valid_target_acc", min_delta=0.0000, patience=100, mode="max")

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        progress_bar = TQDMProgressBar(0) #PB_FRESH

        ### Set the lightning trainer.
        trainer = pl.Trainer(
            min_epochs=0,
            max_epochs=cfg["optim"]["max_epochs"],
            # resume_from_checkpoint=last_checkpoint_file,
            # accelerator="gpu" if args.devices != 0 else "cpu",
            accelerator="gpu",
            # devices=args.devices if args.devices != 0 else "auto",
            devices="auto",
            logger=tb_logger,
            # fast_dev_run=cfg.OUTPUT.FAST_DEV_RUN,
            callbacks=[lr_monitor, checkpoint_callback, progress_bar],
        )

        ### Find learning_rate
        # lr_finder = trainer.tuner.lr_find(model, max_lr=0.1, min_lr=1e-6)
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        # logging.info(lr_finder.suggestion())

        ### Training/validation process
        trainer.fit(model, train_loader, val_loader)

        # # Load the best model
        # best_model_path = checkpoint_callback.best_model_path
        # best_model = get_model(cfg, dataset_train, 1).load_from_checkpoint(best_model_path)
        #
        # # Test process
        # trainer.test(best_model, test_dataloader)

        ### Test process
        trainer.test(model, test_loader)


if __name__ == "__main__":
    main()