from kale.pipeline.base_nn_trainer import BaseNNTrainer
from torch import nn
import torch
import torch.distributed as dist

class LEFTNetTrainer(BaseNNTrainer):
    def __init__(self, model, max_epochs, init_lr=0.001, optimizer_params=None, adapt_lr=False):
        super().__init__(optimizer_params, max_epochs, init_lr, adapt_lr)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch, split_name="valid"):
        y = batch.y
        y_hat = self.forward(batch)
        loss = nn.MSELoss()(y_hat, y)
        log_metrics = {f"{split_name}_loss": loss}
        return loss, log_metrics

        # preds = self.forward(batch)
        # targets = batch.y
        # preds_force = self.model



#
# class DDPLoss(nn.Module):
#     def __init__(self, loss_fn, reduction="mean"):
#         super().__init__()
#         self.loss_fn = loss_fn
#         self.loss_fn.reduction = "sum"
#         self.reduction = reduction
#         assert reduction in ["mean", "sum"]
#
#     def forward(
#         self,
#         input: torch.Tensor,
#         target: torch.Tensor,
#         natoms: torch.Tensor = None,
#         batch_size: int = None,
#     ):
#         # zero out nans, if any
#         found_nans_or_infs = not torch.all(input.isfinite())
#         if found_nans_or_infs is True:
#             logging.warning("Found nans while computing loss")
#             input = torch.nan_to_num(input, nan=0.0)
#
#         if natoms is None:
#             loss = self.loss_fn(input, target)
#         else:  # atom-wise loss
#             loss = self.loss_fn(input, target, natoms)
#         if self.reduction == "mean":
#             num_samples = (
#                 batch_size if batch_size is not None else input.shape[0]
#             )
#             # num_samples = distutils.all_reduce(
#             #     num_samples, device=input.device
#             # )
#             # Multiply by world size since gradients are averaged
#             # across DDP replicas
#             return loss * dist.get_world_size() / num_samples
#         else:
#             return loss