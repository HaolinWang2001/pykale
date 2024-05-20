from kale.pipeline.base_nn_trainer import BaseNNTrainer
from torch import nn
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