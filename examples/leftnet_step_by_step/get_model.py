import logging

from examples.leftnet_step_by_step.m2models.models.leftnet import LEFTNet
from examples.leftnet_step_by_step.m2models.trainers.PyKale_Energy import PykaleEnergyTrainer
from m2models.common import distutils, registry
from m2models.common.data_parallel import OCPDataParallel
from torch.nn.parallel.distributed import DistributedDataParallel


def load_model(config, dataset, num_targets):
    num_atoms = 1
    bond_feat_dim = config["model"].get("num_gaussians", 50)
    model = LEFTNet(
        num_atoms=num_atoms, # not used
        bond_feat_dim = bond_feat_dim, # not used
        num_targets=num_targets, # not used
        otf_graph=False,
        use_pbc=config["model"]["use_pbc"],
        regress_forces=config["model"]["regress_forces"],
        output_dim=config["model"]["output_dim"],
        cutoff=config["model"]["cutoff"],
        num_layers=config["model"]["num_layers"],
        num_radial=config["model"]["num_radial"],
    )
    trainer = PykaleEnergyTrainer(
        # task=config["task"],
        model=model,
        # dataset=config["dataset"],
        max_epochs=config["optim"]["max_epochs"],
        init_lr=config["optim"]["lr_initial"],
        optimizer_params=config["optim"],
        adapt_lr=False,
        run_dir=config.get("run_dir", "./"),
        logger=config.get("logger", "tensorboard"),
    )

    return trainer
# def load_model(config, train_loader, val_loader, test_loader, device, num_targets, logger):
#     # Build model
#     if distutils.is_master():
#         logging.info(f"Loading model: {config['model']}")
#
#     # TODO: deprecated, remove.
#     bond_feat_dim = None
#     bond_feat_dim = config["model"].get("num_gaussians", 50)
#
#     loader = train_loader or val_loader or test_loader
#     model = LEFTNet(
#         loader.dataset[0].x.shape[-1]
#         if loader
#            and hasattr(loader.dataset[0], "x")
#            and loader.dataset[0].x is not None
#         else None,
#         bond_feat_dim,
#         num_targets,
#         # **config["model_attributes"],
#     ).to(device)
#
#     # if distutils.is_master():
#     #     logging.info(
#     #         f"Loaded {model.__class__.__name__} with "
#     #         f"{model.num_params} parameters."
#     #     )
#     #
#     # if logger is not None:
#     #     logger.watch(model)
#     #
#     # model = OCPDataParallel(
#     #     model,
#     #     output_device=device,
#     #     num_gpus=1 if not config["cpu"] else 0,
#     # )
#     # if distutils.initialized() and not config["noddp"]:
#     #     model = DistributedDataParallel(
#     #         model, device_ids=[device]
#     #     )
#
#     trainer = PykaleEnergyTrainer(
#         task=config["task"],
#         model=model,
#         dataset=config["dataset"],
#         max_epochs=config["optim"]["max_epochs"],
#         init_lr=config["optim"]["lr_initial"],
#         optimizer_params=config["optim"],
#         adapt_lr=False,
#         run_dir=config.get("run_dir", "./"),
#         logger=config.get("logger", "tensorboard"),
#     )
#
#     return trainer