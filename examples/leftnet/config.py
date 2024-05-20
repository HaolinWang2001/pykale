"""
Default configurations for action recognition domain adaptation
"""

import os

from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "data"  # "/shared/tale2/Shared"
_C.DATASET.SOURCE = "omdb"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.TRAIN = "data/jarvis/qmof:bandgap/random_train"
_C.DATASET.VALID = "data/jarvis/qmof:bandgap/random_valid"
_C.DATASET.TEST = "data/jarvis/qmof:bandgap/random_test"
_C.DATASET.TARGET = "omdb"
_C.DATASET.REPEAT = 1  # 1
# ---------------------------------------------------------------------------- #
# Task
# ---------------------------------------------------------------------------- #
_C.TASK = CfgNode()
_C.TASK.DATASET = "lmdb"
_C.TASK.DESCRIPTION = "Regressing the energies"
_C.TASK.TYPE = "regression"
_C.TASK.METRIC = "mae"


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.SEED = 2020
# _C.SOLVER.BASE_LR = 0.01  # Initial learning rate
# _C.SOLVER.MOMENTUM = 0.9
# _C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
# _C.SOLVER.NESTEROV = True
#
# _C.SOLVER.TYPE = "SGD"
# _C.SOLVER.MAX_EPOCHS = 30  # "nb_adapt_epochs": 100,
# # _C.SOLVER.WARMUP = True
# _C.SOLVER.MIN_EPOCHS = 5  # "nb_init_epochs": 20,
# _C.SOLVER.TRAIN_BATCH_SIZE = 16  # 150
# # _C.SOLVER.TEST_BATCH_SIZE = 32  # No difference in ADA
#
# # Adaptation-specific solver config
# _C.SOLVER.AD_LAMBDA = True
# _C.SOLVER.AD_LR = True
# _C.SOLVER.INIT_LAMBDA = 1.0
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.EVAL_BATCH_SIZE = 8
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.LR_INITIAL = 0.0005
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.LR_MILESTONES = [5000000000]
_C.SOLVER.WARMUP_STEPS = -1
_C.SOLVER.WARMUP_FACTOR = 1.0
_C.SOLVER.MIN_EPOCHS = 1
_C.SOLVER.MAX_EPOCHS = 10
_C.SOLVER.EVAL_EVERY = 500

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.NAME = "leftnet"
_C.MODEL.CUTOFF = 6.0
_C.MODEL.HIDDEN_CHANNELS = 128
_C.MODEL.NUM_LAYERS = 4
_C.MODEL.NUM_RADIAL = 32
_C.MODEL.REGRESS_FORCES = False
_C.MODEL.USE_PBC = True
_C.MODEL.OTF_GRAPH = False
# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
# _C.DAN = CfgNode()
# _C.DAN.METHOD = "DANN"  # options=["CDAN", "CDAN-E", "DANN", "DAN"]
# _C.DAN.USERANDOM = False
# _C.DAN.RANDOM_DIM = 1024
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CfgNode()
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.FAST_DEV_RUN = False  # True for debug
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.OUT_DIR = os.path.join("outputs", _C.DATASET.SOURCE + "2" + _C.DATASET.TARGET)


def get_cfg_defaults():
    return _C.clone()
