# model_transferlearning.py

# required imports
import torch
from torch import nn
import pytorch_lightning as pl

import wandb
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall


class TransferLearningModule(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        # Model and hyperparameters
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_preds_epoch = []
        self.val_targets_epoch = []

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.criterion(prediction.squeeze(), y.float())

        preds = prediction.squeeze()
        self.train_accuracy.update(preds, y.int())

        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.criterion(prediction.squeeze(), y.float())

        preds = prediction.squeeze()
        self.val_accuracy.update(preds, y.int())
        self.val_preds_epoch.append(preds.detach().cpu())
        self.val_targets_epoch.append(y.detach().cpu())
        self.val_precision.update(preds, y.int())
        self.val_recall.update(preds, y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.criterion(prediction.squeeze(), y.float())
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_targets_epoch).numpy().flatten()
        y_probas = torch.cat(self.val_preds_epoch).numpy().reshape(-1, 1)

        assert y_true.ndim == 1, f"y_true shape: {y_true.shape}"
        assert y_probas.ndim == 2 and y_probas.shape[1] == 1, f"y_probas shape: {y_probas.shape}"

        # Log to wandb
        wandb.log({
            "PR Curve": wandb.plot.pr_curve(y_true, [(1 - x, x) for x in y_probas]),
        })

        # Clean up for next epoch
        self.val_preds_epoch = []
        self.val_targets_epoch = []

        # Reset TorchMetrics
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()