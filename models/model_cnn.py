import torch
import torch.nn as nn
import pytorch_lightning as pl

import wandb
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 2 if downsample else 1

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Identity shortcut
        self.skip = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv_block(x)
        out += identity
        return self.relu(out)

class CnnModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__()

        # Model and hyperparameters
        self.learning_rate = learning_rate          # Hyperparameter tuned by optuna
        self.optimizer_name = optimizer_name        # Hyperparameter tuned by optuna
        self.weight_decay = weight_decay            # Hyperparameter tuned by optuna
        self.scheduler_name = scheduler_name        # Hyperparameter tuned by optuna

        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.save_hyperparameters(ignore=['model'])

        # CNN Model
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(num_features=32), 
            nn.ReLU(), 
            nn.MaxPool2d(2),     
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
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
        
        self.val_probs_epoch = []
        self.val_preds_epoch = []
        self.val_targets_epoch = []

        self.test_accuracy = BinaryAccuracy()
        self.test_preds_epoch = []
        self.test_targets_epoch = []
        self.test_probs_epoch = []     

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
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            return [optimizer], [scheduler]
        elif self.scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after passing through the model.
        This method defines the forward pass of the model, which processes the input tensor through the CNN layers.
        The output is the raw logits from the final layer.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step for the model.
        Args:
            batch (Tuple[Tensor, Tensor]): A tuple containing input data (x) and target labels (y).
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: The computed loss for the current batch.
        This method computes the model's predictions, calculates the loss using the specified criterion,
        updates the training accuracy metric, and logs the loss and accuracy values.
        
        Terminology:
            - logits: The raw, unnormalized outputs from the model's final layer (before applying activation functions like softmax or sigmoid).
            - probs: The probabilities obtained by applying an activation function (e.g., softmax for multi-class, sigmoid for binary) to the logits.
            - preds: The predicted class labels, typically determined by taking the argmax (for multi-class) or thresholding (for binary) on the probs.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(), y.float())

        probs = self.sigmoid(logits.squeeze())
        preds = (probs > 0.5).long()
        # preds = prediction.squeeze()

        self.train_accuracy.update(preds, y.int())
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step during model evaluation.
        Args:
            batch (tuple): A tuple containing input data (x) and target labels (y).
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: The computed loss for the current validation batch.
        Functionality:
            - Computes model predictions (logits) for the input batch.
            - Calculates the loss between predictions and targets.
            - Applies sigmoid activation and thresholding to obtain binary predictions.
            - Updates validation metrics: accuracy, precision, and recall.
            - Stores predictions and targets for the current epoch.
            - Logs loss and metrics for monitoring.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(), y.float())

        probs = self.sigmoid(logits.squeeze())
        preds = (probs > 0.5).long()

        self.val_accuracy.update(preds, y.int())
        self.val_probs_epoch.append(probs.detach().cpu())
        self.val_preds_epoch.append(preds.detach().cpu())
        self.val_targets_epoch.append(y.detach().cpu())
        self.val_precision.update(preds, y.int())
        self.val_recall.update(preds, y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.
        This method performs the following actions:
        - Concatenates and flattens the collected validation targets and predictions for the epoch.
        - Asserts the correct shapes for the true labels and predicted probabilities.
        - Logs the Precision-Recall (PR) curve to Weights & Biases (wandb) for monitoring model performance.
        - Resets the lists storing predictions and targets for the next epoch.
        - Resets the TorchMetrics objects for accuracy, precision, and recall to clear their internal states.
        Assumes:
            - self.val_targets_epoch: List of torch.Tensor containing true labels for the validation set.
            - self.val_preds_epoch: List of torch.Tensor containing predicted probabilities for the validation set.
            - self.val_accuracy, self.val_precision, self.val_recall: TorchMetrics objects with a reset() method.
        """
        # Must be Sequence[numbers.Number]
        y_true = torch.cat(self.val_targets_epoch).cpu().numpy().astype(int).flatten()
        assert y_true.ndim == 1, f"y_true shape: {y_true.shape}"
        assert set(y_true).issubset({0, 1}), f"Unexpected class values in y_true: {set(y_true)}"
        
        # Must be Sequence[Sequence[float]]
        y_probas = torch.cat(self.val_probs_epoch).cpu().numpy().reshape(-1, 1)
        assert y_probas.ndim == 2, f"y_probas shape: {y_probas.shape}"

        y_preds = torch.cat(self.val_preds_epoch).cpu().numpy().reshape(-1, 1)
        
        # Stack the probabilities to get prbabilities for "both" classes
        y_probas_stacked = np.concatenate([1 - y_probas, y_probas], axis=1)
        assert y_probas_stacked.ndim == 2, f"y_probas_stacked shape: {y_probas_stacked.shape}"

        fig, ax = plt.subplots(figsize=(8, 8))
        cm = confusion_matrix(y_true, y_preds.flatten())
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title("Validation Data Confusion Matrix")
        plt.close(fig)  # Verhindert Anzeige im Notebook

        # Log to wandb
        wandb.log({
            "Validation Data PR Curve": wandb.plot.pr_curve(y_true, y_probas_stacked, labels=["cat", "dog"]),
            "Validation Data ROC Curve": wandb.plot.roc_curve(y_true, y_probas_stacked, labels=["cat", "dog"]),
            "Validation Data ROC AUC": roc_auc_score(y_true, y_probas),
            "Validation Data Confusion Matrix": wandb.Image(fig)
        })

        del fig  # Clean up the figure to free memory

        # Clean up for next epoch
        self.val_probs_epoch.clear()
        self.val_preds_epoch.clear()
        self.val_targets_epoch.clear()

        # Reset TorchMetrics
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()

        # y_true = torch.cat(self.val_targets_epoch).numpy().flatten()
        # y_probas = torch.cat(self.val_preds_epoch).numpy().reshape(-1, 1)

        # assert y_true.ndim == 1, f"y_true shape: {y_true.shape}"
        # assert y_probas.ndim == 2 and y_probas.shape[1] == 1, f"y_probas shape: {y_probas.shape}"

        # # Log to wandb
        # wandb.log({
        #     "PR Curve": wandb.plot.pr_curve(y_true, [(1 - x, x) for x in y_probas]),
        # })

        # # Clean up for next epoch
        # self.val_preds_epoch = []
        # self.val_targets_epoch = []

        # # Reset TorchMetrics
        # self.val_accuracy.reset()
        # self.val_precision.reset()
        # self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step during model evaluation.
        Args:
            batch (tuple): A tuple containing input data (x) and target labels (y).
            batch_idx (int): Index of the current batch.
        Returns:
            dict: A dictionary containing the loss, predictions, probabilities, and targets for the current batch.
        Functionality:
            - Computes model outputs and loss for the given batch.
            - Calculates probabilities and binary predictions using a sigmoid activation and thresholding at 0.5.
            - Appends predictions, targets, and probabilities to epoch-level lists for later aggregation.
            - Logs test loss and accuracy metrics.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(), y.float())
        
        probs = self.sigmoid(logits.squeeze())
        preds = (probs >= 0.5).long()

        self.test_preds_epoch.append(preds.detach().cpu())
        self.test_targets_epoch.append(y.detach().cpu())
        self.test_probs_epoch.append(probs.detach().cpu())

        # Log or store for confusion matrix or ROC later 
        # self.log('test_loss', loss)
        self.log_dict({
            'test_loss': loss,
            'test_acc': self.test_accuracy(preds, y)
        }, prog_bar=True)

        return {"loss": loss, "preds": preds, "probs": probs, "targets": y}

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to evaluate and visualize model performance.
        This method performs the following actions:
        - Concatenates and converts the stored predictions, probabilities, and targets for the test epoch.
        - Computes and displays the confusion matrix using scikit-learn.
        - Calculates the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) score.
        - Plots the ROC curve for visual inspection of model performance.
        Assumes that `self.test_preds_epoch`, `self.test_probs_epoch`, and `self.test_targets_epoch`
        are lists of tensors containing predictions, probabilities, and targets for the test epoch, respectively.
        """
        print(f"Predictions shape") 

        preds = torch.cat(self.test_preds_epoch).numpy()
        probs = torch.cat(self.test_probs_epoch).numpy()
        targets = torch.cat(self.test_targets_epoch).numpy()

        # Calculate and plot confusion matrix using sklearn
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()

        # Calculate and plot ROC curve
        fpr, tpr, thresholds = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.title("ROC Curve")
        plt.show()

class KaninchenModel(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        """
        Initializes the KaninchenModel with a specific learning rate.
        Args:
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
        """
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing

        # CNN Model
        # This model is designed for input tensors of size (3, 128, 128)
        # Define a helper function for a Conv-BN-ReLU-MaxPool block
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(pool_kernel)
                )

        #scaling_factor = 2  # Adjusting for input size of (3, 128, 128)
        self.conv1 = conv_block(3, 64)            # Input: (3, 128, 128) -> Output: (64, 64, 64)
        self.conv2 = conv_block(64, 128)          # Output: (128, 32, 32)
        self.conv3 = conv_block(128, 256)         # Output: (256, 16, 16)
        self.conv4 = conv_block(256, 512)         # Output: (512, 8, 8)
        self.conv5 = conv_block(256, 512)  

        self.cnn = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            # self.conv5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.model = nn.Sequential(
            self.cnn,
            self.classifier
        )

    def forward(self, x):
        return self.model(x)
    
class KaninchenModelResidual(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        """
        Initializes the KaninchenModel with a specific learning rate.
        Args:
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
        """
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing

        # CNN Model
        # This model is designed for input tensors of size (3, 128, 128)
        # Define a helper function for a Conv-BN-ReLU-MaxPool block
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(pool_kernel)
                )

        #scaling_factor = 2  # Adjusting for input size of (3, 128, 128)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )

        self.layer1 = conv_block(64, 128)              # Input: (3, 128, 128) # -> Output: (64, 64, 64)
        self.layer2 = conv_block(128, 128)             # Output: (128, 32, 32)
        self.layer3 = conv_block(128, 256)             # Output: (256, 16, 16)
        self.layer4 = ResidualBlock(256, 256, downsample=True)       # Output: (256, 8, 8)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256*8*8, 1)  # Binary classification

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
#SiLU
class KaninchenModel_v1(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        """
        Initializes the KaninchenModel with a specific learning rate.
        Args:
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
        """
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing

        # CNN Model
        # This model is designed for input tensors of size (3, 128, 128)
        # Define a helper function for a Conv-BN-ReLU-MaxPool block
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.SiLU(),
                nn.MaxPool2d(pool_kernel)
                )

        #scaling_factor = 2  # Adjusting for input size of (3, 128, 128)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            )

        self.layer1 = conv_block(64, 128)              # Input: (3, 128, 128) # -> Output: (64, 64, 64)
        self.layer2 = conv_block(128, 128)             # Output: (128, 32, 32)
        self.layer3 = conv_block(128, 256)             # Output: (256, 16, 16)
        self.layer4 = ResidualBlock(256, 256, downsample=True)       # Output: (256, 8, 8)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256*8*8, 1)  # Binary classification

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# Varied out_channels
class KaninchenModel_v2(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        """
        Initializes the KaninchenModel with a specific learning rate.
        Args:
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
        """
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing

        # CNN Model
        # This model is designed for input tensors of size (3, 128, 128)
        # Define a helper function for a Conv-BN-ReLU-MaxPool block
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(pool_kernel)
                )

        #scaling_factor = 2  # Adjusting for input size of (3, 128, 128)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3,128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        
        self.layer1 = conv_block(128, 128)        
        self.layer2 = conv_block(128, 256)
        self.layer3 = conv_block(256, 256)  
        self.layer4 = conv_block(256, 512)  
        self.layer5 = ResidualBlock(512, 512, downsample=True)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 1)

    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# dense block
class KaninchenModel_v3(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_kernel)
            )

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = conv_block(64, 128)              # (64, 128,128) -> (128, 64,64)
        self.layer2 = conv_block(128, 128)             # (128, 64,64) -> (128, 32,32)
        self.layer3 = conv_block(128, 256)             # (128, 32,32) -> (256, 16,16)
        self.layer4 = ResidualBlock(256, 256, downsample=True)   # (256, 16,16) -> (256, 8,8)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (256, 1, 1) 
        self.flatten = nn.Flatten()

        # Dense block
        self.fc1 = nn.Linear(256, 128)        #A MODIFIER
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.fc_out(x)
        return x

# dropout added
class KaninchenModel_v4(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(pool_kernel),
                nn.Dropout(0.25)
            )

        def final_block():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),  # Matches your expected Flatten shape
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()  # For binary classification
            )
        

        self.layer1 = conv_block(3, 32)     # Input: (3, 128, 128) -> (32, 64, 64)
        self.layer2 = conv_block(32, 64)    # -> (64, 32, 32)
        self.layer3 = conv_block(64, 128)   # -> (128, 16, 16)
        self.classifier = final_block()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x   
    
#focus on presence not position
class KaninchenModel_v5(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        # Depthwise Separable Convolution Block with Downsampling via stride
        def ds_conv_block(in_channels, out_channels, downsample=True):
            stride = 2 if downsample else 1
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                # Pointwise
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Dropout(0.25)
            )

        self.block1 = ds_conv_block(3, 64)      # (3,128,128) -> (64,64,64)
        self.block2 = ds_conv_block(64, 128)    # -> (128,32,32)
        self.block3 = ds_conv_block(128, 256)   # -> (256,16,16)
        self.block4 = ds_conv_block(256, 256)   # -> (256,8,8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (256,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x