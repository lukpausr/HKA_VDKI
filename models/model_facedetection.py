import torch
import pytorch_lightning as pl
import timm
import wandb

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

# from .model_transferlearning import TransferLearningModule

def getEfficientNetB4_model(amount_of_trainable_linear_layers=1, num_classes=6):
    """
    Function to get the EfficientNet B4 model with pretrained weights.
    Returns:
        model: A PyTorch model instance of EfficientNet B4.
    """
    # Load the EfficientNet B4 model with pretrained weights
    model = timm.create_model('efficientnet_b4', pretrained=True)
    
    # Modify the classifier for binary classification
    # num_classes = len(config['name_list'])
    # num_classes = 6
    if amount_of_trainable_linear_layers == 1:
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif amount_of_trainable_linear_layers == 2:
        # If two linear layers are trainable, we add an intermediate layer
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),  # Add dropout for regularization
            torch.nn.Linear(model.classifier.in_features, 256),  # Intermediate layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Dropout(p=0.2),  # Another dropout layer
            torch.nn.Linear(256, num_classes)
        )
    
    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model, "FD_EfficientNetB4"

def getConvNextV2_model(amount_of_trainable_linear_layers=1, num_classes=6):

    # Load the EfficientNet B3 model with pretrained weights
    model = timm.create_model('convnextv2_base', pretrained=True)
    
    # Modify the classifier for binary classification
    if amount_of_trainable_linear_layers == 1:
        model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes)
    elif amount_of_trainable_linear_layers == 2:
        model.head.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),                                # Add dropout for regularization
            torch.nn.Linear(model.head.fc.in_features, 256),    # Intermediate layer
            torch.nn.ReLU(),                                        # Activation function
            torch.nn.Dropout(p=0.2),                                # Another dropout layer
            torch.nn.Linear(256, num_classes)
        )
    
    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.fc.parameters():
        param.requires_grad = True
    
    return model, "FDund ve_ConvNextV2_base"

class TransferLearningModuleMulticlass(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name

        self.num_classes = num_classes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)

        self.save_hyperparameters(ignore=['model'])

        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=num_classes, average='macro')

        self.val_probs_epoch = []
        self.val_preds_epoch = []
        self.val_targets_epoch = []

        self.test_preds_epoch = []
        self.test_targets_epoch = []
        self.test_probs_epoch = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        Configures and returns the optimizer for training the model.

        Returns:
            torch.optim.Optimizer: An Adam optimizer initialized with the model's parameters and the specified learning rate.
        """
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            return [optimizer], [scheduler]
        elif self.scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer
        # if self.optimizer_name == 'Adam':
        #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # elif self.optimizer_name == 'SGD':
        #     optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # else:
        #     raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # if self.scheduler_name == 'StepLR':
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        #     return [optimizer], [scheduler]
        # else:
        #     return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y_indices)
        probs = self.softmax(logits)
        preds = torch.argmax(probs, dim=1)
        self.train_accuracy.update(preds, y_indices)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y_indices)
        probs = self.softmax(logits)  # (batch_size, num_classes)
        preds = torch.argmax(probs, dim=1)  # (batch_size,)
        self.val_accuracy.update(preds, y_indices)
        self.val_probs_epoch.append(probs.detach().cpu())
        self.val_preds_epoch.append(preds.detach().cpu())
        self.val_targets_epoch.append(y_indices.detach().cpu())
        self.val_precision.update(preds, y_indices)
        self.val_recall.update(preds, y_indices)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_targets_epoch).cpu().numpy().astype(int).flatten()
        y_probas = torch.cat(self.val_probs_epoch).cpu().numpy()
        y_preds = torch.cat(self.val_preds_epoch).cpu().numpy().flatten()

        fig, ax = plt.subplots(figsize=(8, 8))
        cm = confusion_matrix(y_true, y_preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title("Validation Data Confusion Matrix")
        plt.close(fig)

        class_labels = [f"Class {i}" for i in range(self.num_classes)]

        try:
            wandb.log({
                "Validation Data PR Curve": wandb.plot.pr_curve(y_true, y_probas, labels=class_labels),
                "Validation Data ROC Curve": wandb.plot.roc_curve(y_true, y_probas, labels=class_labels),
                "Validation Data ROC AUC": roc_auc_score(y_true, y_probas, multi_class='ovr'),
                "Validation Data Confusion Matrix": wandb.Image(fig)
            })
        except Exception as e:
            print(f"Warning: Validation metrics could not be calculated/logged to wandb: {e}")

        del fig
        self.val_probs_epoch.clear()
        self.val_preds_epoch.clear()
        self.val_targets_epoch.clear()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=1)
        loss = self.criterion(logits, y_indices)
        probs = self.softmax(logits)
        preds = torch.argmax(probs, dim=1)
        self.test_preds_epoch.append(preds.detach().cpu())
        self.test_targets_epoch.append(y.detach().cpu())
        self.test_probs_epoch.append(probs.detach().cpu())
        self.log_dict({
            'test_loss': loss,
            'test_acc': self.train_accuracy(preds, y_indices)
        }, prog_bar=True)
        return {"loss": loss, "preds": preds, "probs": probs, "targets": y}

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds_epoch).numpy()
        probs = torch.cat(self.test_probs_epoch).numpy()
        targets = torch.cat(self.test_targets_epoch).numpy()
        # Convert one-hot/multilabel to class labels if needed
        # preds = np.argmax(preds, axis=1)
        targets = np.argmax(targets, axis=1)

        # Plot Confusion Matrix
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()

class TL_ConvNextV2(TransferLearningModuleMulticlass):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR', amount_of_trainable_linear_layers=1, num_classes=6):
        model, model_name = getConvNextV2_model(amount_of_trainable_linear_layers, num_classes)
        super().__init__(model, num_classes, learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.model_name = model_name
        self.save_hyperparameters()  # Save hyperparameters for logging

class TL_EfficientNetB4(TransferLearningModuleMulticlass):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR', amount_of_trainable_linear_layers=1, num_classes=6):
        model, model_name = getEfficientNetB4_model(amount_of_trainable_linear_layers, num_classes)
        super().__init__(model, num_classes, learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.model_name = model_name
        self.save_hyperparameters()  # Save hyperparameters for logging