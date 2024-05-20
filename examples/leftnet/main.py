"""This example is about domain adaptation for action recognition, using PyTorch Lightning.

Reference: https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
"""

import argparse
import logging

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from config import get_cfg_defaults
from model import get_model
from data import get_data
from data import Data
from data import *
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

# embeddings
from qmof_khot_embeddings import QMOF_KHOT_EMBEDDINGS
from khot_embeddings import KHOT_EMBEDDINGS


from kale.loaddata.video_access import VideoDataset
from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets
from kale.utils.seed import set_seed
from lmdb_data import LmdbDataset
import numpy as np
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
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # ---- setup output ----
    format_str = "@%(asctime)s %(name)s [%(levelname)s] - (%(message)s)"
    logging.basicConfig(format=format_str)
    # ---- setup dataset ----
    seed = cfg.SOLVER.SEED
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
    dataset_train = LmdbDataset({"src" : cfg.DATASET.TRAIN})
    dataset_valid = LmdbDataset({"src" : cfg.DATASET.VALID})
    dataset_test = LmdbDataset({"src" : cfg.DATASET.TEST})
    # joined_data = Data('qmof', dataset_train, dataset_valid, dataset_test)
    # result = joined_data.join_dataset()

    dataset_full = JoinedLmdbDataset([dataset_train, dataset_valid, dataset_test])

    embeddings = KHOT_EMBEDDINGS
    # dataset_full_with_embedding = DatasetWithEmbedding(dataset_full, embeddings)

    # data_loader = DataLoader(dataset_full, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=cfg.SOLVER.NUM_WORKERS)

    train_dataloader = DataLoader(dataset_train, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.SOLVER.NUM_WORKERS)
    valid_dataloader = DataLoader(dataset_valid, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                  num_workers=cfg.SOLVER.NUM_WORKERS)
    test_dataloader = DataLoader(dataset_test, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.SOLVER.NUM_WORKERS)

    # ---- training/test process ----
    ### Repeat multiple times to get std
    for i in range(0, cfg.DATASET.REPEAT):
        seed = seed + i * 10
        set_seed(seed)  # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        print(f"==> Building model for seed {seed} ......")
        # ---- setup model and logger ----
        model = get_model(cfg, dataset_train, 2)
        tb_logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.OUT_DIR, name="seed{}".format(seed))
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
        progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

        ### Set the lightning trainer.
        trainer = pl.Trainer(
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            # resume_from_checkpoint=last_checkpoint_file,
            accelerator="gpu" if args.devices != 0 else "cpu",
            devices=args.devices if args.devices != 0 else "auto",
            logger=tb_logger,
            fast_dev_run=cfg.OUTPUT.FAST_DEV_RUN,
            callbacks=[lr_monitor, checkpoint_callback, progress_bar],
        )

        ### Find learning_rate
        # lr_finder = trainer.tuner.lr_find(model, max_lr=0.1, min_lr=1e-6)
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        # logging.info(lr_finder.suggestion())

        ### Training/validation process
        trainer.fit(model, train_dataloader, valid_dataloader)

        ### Test process
        trainer.test()


if __name__ == "__main__":
    main()
