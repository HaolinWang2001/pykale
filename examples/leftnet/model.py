# # =============================================================================
# # Author: Xianyuan Liu, xianyuan.liu@outlook.com
# #         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# # =============================================================================
#
# """
# Define the learning model and configure training parameters.
# References from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
# """
#
# from copy import deepcopy
#
# from kale.embed.video_feature_extractor import get_video_feat_extractor
# from kale.pipeline import domain_adapter, video_domain_adapter
# from kale.predict.class_domain_nets import ClassNetVideo, DomainNetVideo
#
#
# def get_config(cfg):
#     """
#     Sets the hyper parameter for the optimizer and experiment using the config file
#
#     Args:
#         cfg: A YACS config object.
#     """
#
#     config_params = {
#         "train_params": {
#             "batch_size": cfg.SOLVER.BATCH_SIZE,
#             "eval_batch_size": cfg.SOLVER.EVAL_BATCH_SIZE,
#             "num_workers": cfg.SOLVER.NUM_WORKERS,
#             "lr_initial": cfg.SOLVER.LR_INITIAL,
#             "lr_gamma": cfg.SOLVER.LR_GAMMA,
#             "lr_milestones": cfg.SOLVER.LR_MILESTONES,
#             "warmup_steps": cfg.SOLVER.WARMUP_STEPS,
#             "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
#             "max_epochs": cfg.SOLVER.MAX_EPOCHS,
#             "eval_every": cfg.SOLVER.EVAL_EVERY,
#         },
#         "data_params": {
#             "dataset_train": cfg.DATASET.TRAIN,
#             "dataset_valid": cfg.DATASET.VALID,
#             "dataset_test": cfg.DATASET.TEST,
#         },
#     }
#     return config_params
#
#
# # Based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
# def get_model(cfg, dataset, num_classes):
#     """
#     Builds and returns a model and associated hyperparameters according to the config object passed.
#
#     Args:
#         cfg: A YACS config object.
#         dataset: A multi domain dataset consisting of source and target datasets.
#         num_classes: The class number of specific dataset.
#     """
#
#     # setup feature extractor
#     feature_network, class_feature_dim, domain_feature_dim = get_video_feat_extractor(
#         cfg.MODEL.METHOD.upper(), cfg.DATASET.IMAGE_MODALITY, cfg.MODEL.ATTENTION, num_classes
#     )
#     # setup classifier
#     classifier_network = ClassNetVideo(input_size=class_feature_dim, n_class=num_classes)
#
#     config_params = get_config(cfg)
#     train_params = config_params["train_params"]
#     train_params_local = deepcopy(train_params)
#     method_params = {}
#
#     method = domain_adapter.Method(cfg.DAN.METHOD)
#
#     critic_input_size = domain_feature_dim
#     # setup critic network
#
#     # when CDAN
#     # critic_input_size = domain_feature_dim * num_classes
#     critic_network = DomainNetVideo(input_size=critic_input_size)
#
#     # The following calls kale.loaddata.dataset_access for the first time
#     model = video_domain_adapter.create_dann_like_video(
#         method=method,
#         dataset=dataset,
#         image_modality=cfg.DATASET.IMAGE_MODALITY,
#         feature_extractor=feature_network,
#         task_classifier=classifier_network,
#         critic=critic_network,
#         **method_params,
#         **train_params_local,
#     )
#
#     return model, train_params
from examples.leftnet.leftnet import LEFTNet
from copy import deepcopy

from examples.leftnet.trainer import LEFTNetTrainer


def get_config(cfg):
    """
    Sets the hyperparameters for the optimizer and experiment using the config file

    Args:
        cfg: A YACS config object.
    """

    config_params = {
        "train_params": {
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "num_workers": cfg.SOLVER.NUM_WORKERS,
            "lr_initial": cfg.SOLVER.LR_INITIAL,
            "lr_gamma": cfg.SOLVER.LR_GAMMA,
            "lr_milestones": cfg.SOLVER.LR_MILESTONES,
            "warmup_steps": cfg.SOLVER.WARMUP_STEPS,
            "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
            "max_epochs": cfg.SOLVER.MAX_EPOCHS,
            "eval_every": cfg.SOLVER.EVAL_EVERY,
        },
        "model_attributes": {
            "cutoff": cfg.MODEL.CUTOFF,
            "hidden_channels": cfg.MODEL.HIDDEN_CHANNELS,
            "num_layers": cfg.MODEL.NUM_LAYERS,
            "num_radial": cfg.MODEL.NUM_RADIAL,
            "regress_forces": cfg.MODEL.REGRESS_FORCES,
            "use_pbc": cfg.MODEL.USE_PBC,
            "otf_graph": cfg.MODEL.OTF_GRAPH,
        },
        "data_params": {
            "root": cfg.DATASET.ROOT,
            "source": cfg.DATASET.SOURCE,
            "train": cfg.DATASET.TRAIN,
            "valid": cfg.DATASET.VALID,
            "test": cfg.DATASET.TEST,
            "target": cfg.DATASET.TARGET,
            # "repeat": cfg.DATASET.REPEAT,
        },
    }
    return config_params



def get_model(cfg, dataset, num_classes):
    """
    Builds and returns the LEFTNet model according to the config object passed.

    Args:
        cfg: A YACS config object.
    """
    # Extract parameters from the config
    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    model_attributes = config_params["model_attributes"]
    model_attributes_local = deepcopy(model_attributes)
    method_params = {}

    num_atoms = (
        dataset.x.shape[-1]
        if dataset and hasattr(dataset[0], "x")
            and dataset[0].x is not None
            else None)
    bond_feat_dim = model_attributes_local.get("num_gaussians", 50)
    num_targets = num_classes

    leftnet_model = LEFTNet(
        num_atoms=num_atoms,
        bond_feat_dim=bond_feat_dim,
        num_targets=num_targets,
        **model_attributes
    )

    trainer = LEFTNetTrainer(model=leftnet_model, max_epochs=train_params["max_epochs"], init_lr=train_params["lr_initial"])

    return trainer