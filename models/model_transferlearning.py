import torch
import pytorch_lightning as pl

import wandb
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

import numpy as np

import timm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def getEfficientNetB4_model(amount_of_trainable_linear_layers=2):
    """
    Function to get the EfficientNet B3 model with pretrained weights.
    Returns:
        model: A PyTorch model instance of EfficientNet B3.
    """
    # Load the EfficientNet B3 model with pretrained weights
    model = timm.create_model('efficientnet_b4', pretrained=True)
    
    # Modify the classifier for binary classification
    num_classes = 1  # For binary classification (OK/NOK)
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
    
    return model, "TL_EfficientNetB4"

def getResNet152D_model(amount_of_trainable_linear_layers=2):

    # Load the EfficientNet B3 model with pretrained weights
    model = timm.create_model('resnet152d', pretrained=True)
    
    # Modify the classifier for binary classification
    num_classes = 1  # For binary classification (OK/NOK)
    if amount_of_trainable_linear_layers == 1:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif amount_of_trainable_linear_layers == 2:
        # If two linear layers are trainable, we add an intermediate layer
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),                        # Add dropout for regularization
            torch.nn.Linear(model.fc.in_features, 256),     # Intermediate layer
            torch.nn.ReLU(),                                # Activation function
            torch.nn.Dropout(p=0.2),                        # Another dropout layer
            torch.nn.Linear(256, num_classes)
        )
    
    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model, "TL_ResNet50D"

def getInceptionV4_model(amount_of_trainable_linear_layers=2):

    # Load the EfficientNet B3 model with pretrained weights
    model = timm.create_model('inception_v4', pretrained=True)
    
    # Modify the classifier for binary classification
    num_classes = 1  # For binary classification (OK/NOK)
    if amount_of_trainable_linear_layers == 1:
        model.last_linear = torch.nn.Linear(model.last_linear.in_features, num_classes)
    elif amount_of_trainable_linear_layers == 2:
        model.last_linear = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),                                # Add dropout for regularization
            torch.nn.Linear(model.last_linear.in_features, 256),    # Intermediate layer
            torch.nn.ReLU(),                                        # Activation function
            torch.nn.Dropout(p=0.2),                                # Another dropout layer
            torch.nn.Linear(256, num_classes)
        )
    
    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.last_linear.parameters():
        param.requires_grad = True
    
    return model, "TL_InceptionV4"

def getConvNextV2_model(amount_of_trainable_linear_layers=2):

    # Load the EfficientNet B3 model with pretrained weights
    model = timm.create_model('convnextv2_base', pretrained=True)
    
    # Modify the classifier for binary classification
    num_classes = 1  # For binary classification (OK/NOK)
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
    
    return model, "TL_ConvNextV2_base"

class TransferLearningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__()

        # Model and hyperparameters
        self.model = model

        self.learning_rate = learning_rate          # Hyperparameter tuned by optuna
        self.optimizer_name = optimizer_name        # Hyperparameter tuned by optuna
        self.weight_decay = weight_decay            # Hyperparameter tuned by optuna
        self.scheduler_name = scheduler_name        # Hyperparameter tuned by optuna

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.sigmoid = torch.nn.Sigmoid()

        # self.save_hyperparameters(ignore=['model'])
        self.save_hyperparameters()

        # Metrics
        self.train_accuracy = BinaryAccuracy()
        
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

        self.val_probs_epoch = []
        self.val_preds_epoch = []
        self.val_targets_epoch = []

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
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
            "Validation Data PR Curve": wandb.plot.pr_curve(y_true, y_probas_stacked, labels=["Kleintiere", "Zwergkaninchen"]),
            "Validation Data ROC Curve": wandb.plot.roc_curve(y_true, y_probas_stacked, labels=["Kleintiere", "Zwergkaninchen"]),
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

class TL_ConvNextV2(TransferLearningModule):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR', amount_of_trainable_linear_layers=2):
        model, model_name = getConvNextV2_model(amount_of_trainable_linear_layers)
        super().__init__(model, learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.model_name = model_name
        self.save_hyperparameters()  # Save hyperparameters for logging

class TL_EfficientNetB4(TransferLearningModule):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR', amount_of_trainable_linear_layers=2):
        model, model_name = getEfficientNetB4_model(amount_of_trainable_linear_layers)
        super().__init__(model, learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.model_name = model_name
        self.save_hyperparameters()  # Save hyperparameters for logging

class TL_ResNet152D(TransferLearningModule):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR', amount_of_trainable_linear_layers=2):
        model, model_name = getResNet152D_model(amount_of_trainable_linear_layers)
        super().__init__(model, learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.model_name = model_name
        self.save_hyperparameters()  # Save hyperparameters for logging

class TL_InceptionV4(TransferLearningModule):
    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR', amount_of_trainable_linear_layers=2):
        model, model_name = getInceptionV4_model(amount_of_trainable_linear_layers)
        super().__init__(model, learning_rate, optimizer_name, weight_decay, scheduler_name)
        self.model_name = model_name
        self.save_hyperparameters()  # Save hyperparameters for logging