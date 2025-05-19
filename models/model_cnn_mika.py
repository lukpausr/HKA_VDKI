import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class CatsDogsModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=32), nn.ReLU(), nn.MaxPool2d(2),     
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(num_features=64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(num_features=128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.test_accuracy = BinaryAccuracy()
        self.test_preds_epoch = []
        self.test_targets_epoch = []
        self.test_probs_epoch = []

    def forward(self, x):
        return self.model(x)

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
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.criterion(prediction.squeeze(), y.float())
        self.log('test_loss', loss)
        
        probs = self.sigmoid(prediction.squeeze())
        preds = (probs >= 0.5).long()

        self.test_preds_epoch.append(preds.detach().cpu())
        self.test_targets_epoch.append(y.detach().cpu())
        self.test_probs_epoch.append(probs.detach().cpu())

        # Log or store for confusion matrix or ROC later
        self.log_dict({'test_acc': self.test_accuracy(preds, y)}, prog_bar=True)
        return {"preds": preds, "probs": probs, "targets": y}

    def on_test_epoch_end(self):
        print(f"Predictions shape") 

        preds = torch.cat(self.test_preds_epoch).numpy()
        probs = torch.cat(self.test_probs_epoch).numpy()
        targets = torch.cat(self.test_targets_epoch).numpy()

        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()

        fpr, tpr, thresholds = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.title("ROC Curve")
        plt.show()
