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

        self.layer1 = conv_block(64, 128)              # Input: (64, 128, 128) # -> Output: (128, 64, 64)
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

        # Input: (3, 128, 128) # -> Output: (64, 128, 128)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            )

        self.layer1 = conv_block(64, 128)              # Input: (64, 128, 128) # -> Output: (128, 64, 64)
        self.layer2 = conv_block(128, 128)             # Output: (128, 32, 32)
        self.layer3 = conv_block(128, 256)             # Output: (256, 16, 16)
        self.layer4 = ResidualBlock(256, 256, downsample=True)       # Output: (256, 8, 8)

        self.flatten = nn.Flatten() # Output: (256*8*8)
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
    
# linear layers added and AdaptiveAvgPool2d
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

        self.layer1 = conv_block(64, 128)             
        self.layer2 = conv_block(128, 128)             
        self.layer3 = conv_block(128, 256)             
        self.layer4 = ResidualBlock(256, 256, downsample=True)   

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)        
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
    

# dropout and Sigmoid added 
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
                nn.Linear(128 * 16 * 16, 256),  
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()  
            )
        

        self.layer1 = conv_block(3, 32)     
        self.layer2 = conv_block(32, 64)   
        self.layer3 = conv_block(64, 128)   
        self.classifier = final_block()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x   
    

#dropout and AdaptiveAvgPool2d
class KaninchenModel_v5(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()


        def ds_conv_block(in_channels, out_channels, downsample=True):
            stride = 2 if downsample else 1
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Dropout(0.25)
            )

        self.block1 = ds_conv_block(3, 64)      
        self.block2 = ds_conv_block(64, 128)   
        self.block3 = ds_conv_block(128, 256)  
        self.block4 = ds_conv_block(256, 256)  

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
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
    

######## GEN 2 ########


# out_channels added, linear layers added and AdaptiveAvgPool2d
class KaninchenModel_v6(CnnModel):
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
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer1 = conv_block(128, 128)        
        self.layer2 = conv_block(128, 256)
        self.layer3 = conv_block(256, 256)  
        self.layer4 = conv_block(256, 512)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 128)        
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


# out_channels added and linear layers added
class KaninchenModel_v7(CnnModel):
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
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer1 = conv_block(128, 128)        
        self.layer2 = conv_block(128, 256)
        self.layer3 = conv_block(256, 256)  
        self.layer4 = conv_block(256, 512)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 256)        
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)        
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
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
        x = self.relu_fc3(self.fc3(x))
        x = self.fc_out(x)
        return x
    
    


# SE block
class KaninchenModel_v8(CnnModel):
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

        self.layer1 = conv_block(64, 128)
        self.layer2 = conv_block(128, 128)
        self.layer3 = conv_block(128, 256)
        self.layer4 = ResidualBlock(256, 256, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(256, 1)

        # SE block parameters
        self.se_fc1 = nn.Linear(256, 256 // 16)
        self.se_fc2 = nn.Linear(256 // 16, 256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Squeeze: Global Avg Pool
        
        se = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # Shape: (B, 256)
        se = torch.nn.functional.relu(self.se_fc1(se))
        se = self.sigmoid(self.se_fc2(se)).view(x.size(0), 256, 1, 1)  # Shape: (B, 256, 1, 1)

        # Excite: scale input features
        x = x * se  # Channel-wise multiplication

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc_out(x)
        return x
    


# linear layers added, SilU and dropout 
class KaninchenModel_v9(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(pool_kernel)
            )

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        self.layer1 = conv_block(64, 128)              
        self.layer2 = conv_block(128, 128)             
        self.layer3 = conv_block(128, 256)             
        self.layer4 = ResidualBlock(256, 256, downsample=True)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)        
        self.act1 = nn.SiLU(inplace=True)
        self.dropout1=nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.SiLU(inplace=True)
        self.dropout2=nn.Dropout(0.3)

        self.fc_out = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)


        x = self.act1(self.fc1(x))
        x = self.dropout1(x)    
        x = self.act2(self.fc2(x))
        x = self.dropout2(x)    
        x = self.fc_out(x)
        return x
    

   
# out_channels + v5(dropout and AdaptiveAvgPool2d)
class KaninchenModel_v10(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        # Depthwise Separable Convolution Block with Downsampling via stride
        def ds_conv_block(in_channels, out_channels, downsample=True):
            stride = 2 if downsample else 1
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Dropout(0.25)
            )

        self.block1 = ds_conv_block(3, 128)      
        self.block2 = ds_conv_block(128, 256)   
        self.block3 = ds_conv_block(256, 512)  
        self.block4 = ds_conv_block(512, 512)  

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    

######## GEN 3 ########

# v5(dropout and AdaptiveAvgPool2d) + out_channels + linear layers
class KaninchenModel_v11(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        # Depthwise Separable Convolution Block with Downsampling via stride
        def ds_conv_block(in_channels, out_channels, downsample=True):
            stride = 2 if downsample else 1
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Dropout(0.25)
            )

        self.block1 = ds_conv_block(3, 128)      
        self.block2 = ds_conv_block(128, 256)   
        self.block3 = ds_conv_block(256, 512)  
        self.block4 = ds_conv_block(512, 512)  

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten=nn.Flatten()

        self.fc1 = nn.Linear(512, 256)        
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)        
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.fc_out(x)
        return x
    

# linear layers and AdaptiveAvgPool2d
class KaninchenModel_v12(CnnModel):
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

        self.layer1 = conv_block(64, 128)              
        self.layer2 = conv_block(128, 128)             
        self.layer3 = conv_block(128, 256)             
        self.layer4 = ResidualBlock(256, 256, downsample=True)   

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)        
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, 32)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(32, 1)  

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
        x = self.relu_fc3(self.fc3(x))
        x = self.fc_out(x)
        return x
    


# out_channels + linear layers + GELU
class KaninchenModel_v13(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.MaxPool2d(pool_kernel)
            )

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.layer1 = conv_block(128, 128)        
        self.layer2 = conv_block(128, 256)
        self.layer3 = conv_block(256, 256)  
        self.layer4 = conv_block(256, 512)  
        self.layer5 = ResidualBlock(512, 512, downsample=True)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 256)        
        self.relu_fc1 = nn.GELU()
        self.fc2 = nn.Linear(256, 128)        
        self.relu_fc2 = nn.GELU()
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.GELU()

        self.fc_out = nn.Linear(64, 1)


    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc3(self.fc3(x))
        x = self.fc_out(x)
        return x
    

#v7(out_channels + linear layers) + conv_block(512,512)
class KaninchenModel_v14(CnnModel):
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
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer1 = conv_block(128, 128)        
        self.layer2 = conv_block(128, 256)
        self.layer3 = conv_block(256, 256)  
        self.layer4 = conv_block(256, 512)
        self.layer5 = conv_block(512, 512)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 256)        
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)        
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(64, 1)


    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc3(self.fc3(x))
        x = self.fc_out(x)
        return x


#v7(out_channels + linear layers) + conv_block(512,1024) + conv_block(1024,1024) + AdaptiveAvgPool2d
class KaninchenModel_v15(CnnModel):
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
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer1 = conv_block(128, 128)        
        self.layer2 = conv_block(128, 256)
        self.layer3 = conv_block(256, 256)  
        self.layer4 = conv_block(256, 512)
        self.layer5 = conv_block(512, 1024)
        self.layer6 = conv_block(1024, 1024)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1024, 512)        
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 256)        
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, 128)        
        self.relu_fc3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(128, 64)
        self.relu_fc4 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(64, 1)


    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc3(self.fc3(x))
        x = self.relu_fc4(self.fc4(x))
        x = self.fc_out(x)
        return x


# v14(linear layers + conv_block(512,512)) + conv_block(64, 128) + AdaptiveAvgPool2d
class KaninchenModel_v16(CnnModel):
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

        self.layer1 = conv_block(64, 128)
        self.layer2 = conv_block(128, 128)        
        self.layer3 = conv_block(128, 256)
        self.layer4 = conv_block(256, 256)  
        self.layer5 = conv_block(256, 512)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 256)        
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)        
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(64, 1)


    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc3(self.fc3(x))
        x = self.fc_out(x)
        return x


# out_channels + linear layers + AdaptiveAvgPool2d
class KaninchenModel_v17(CnnModel):
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
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1),
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

        self.fc1 = nn.Linear(512, 256)        
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)        
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.fc_out = nn.Linear(64, 1)


    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc3(self.fc3(x))
        x = self.fc_out(x)
        return x


# dense block + AdaptativeAvgPool2d 
class KaninchenModel_v18(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = conv_block(64, 128)  

        self.dense_layers = nn.ModuleList()
        self.num_dense_layers = 4
        self.growth_rate = 32
        dense_in_channels = 128
        for i in range(self.num_dense_layers):
            self.dense_layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(dense_in_channels + i * self.growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dense_in_channels + i * self.growth_rate, self.growth_rate, kernel_size=3, padding=1)
                )
            )

        self.transition = nn.Sequential(
            nn.BatchNorm2d(dense_in_channels + self.num_dense_layers * self.growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(dense_in_channels + self.num_dense_layers * self.growth_rate, 256, kernel_size=1),
            nn.AvgPool2d(2)
        )

        self.layer4 = ResidualBlock(256, 256, downsample=True)  

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)

        # Inline DenseBlock logic
        dense_outputs = [x]
        for layer in self.dense_layers:
            concat = torch.cat(dense_outputs, dim=1)
            out = layer(concat)
            dense_outputs.append(out)
        x = torch.cat(dense_outputs, dim=1)

        x = self.transition(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.fc_out(x)
        return x


# v3 mit kernel_size=7
class KaninchenModel_v19(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        def conv_block(in_channels, out_channels, kernel_size=7, padding=1, pool_kernel=2):
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
            nn.Conv2d(3, 64, kernel_size=7, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = conv_block(64, 128)              
        self.layer2 = conv_block(128, 128)             
        self.layer3 = conv_block(128, 256)             
        self.layer4 = ResidualBlock(256, 256, downsample=True)   

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)        
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


# v9 kernel_size=5
class KaninchenModel_v20(CnnModel):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.save_hyperparameters()

        def conv_block(in_channels, out_channels, kernel_size=5, padding=1, pool_kernel=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(pool_kernel)
            )

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        self.layer1 = ResidualBlock(64, 128)             
        self.layer2 = ResidualBlock(128, 128)          
        self.layer3 = ResidualBlock(128, 256)           
        self.layer4 = ResidualBlock(256, 256, downsample=True)   

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 128)        
        self.act1 = nn.SiLU(inplace=True)
        self.dropout1=nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.SiLU(inplace=True)
        self.dropout2=nn.Dropout(0.3)

        self.fc_out = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)


        x = self.act1(self.fc1(x))
        x = self.dropout1(x)    
        x = self.act2(self.fc2(x))
        x = self.dropout2(x)    
        x = self.fc_out(x)
        return x


#v3 whithout avgpool
class KaninchenModel_v21(CnnModel):
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

        self.layer1 = conv_block(64, 128)              
        self.layer2 = conv_block(128, 128)             
        self.layer3 = conv_block(128, 256)             
        self.layer4 = ResidualBlock(256, 256, downsample=True)   

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256*8*8, 128)        
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
        x = self.flatten(x)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.fc_out(x)
        return x