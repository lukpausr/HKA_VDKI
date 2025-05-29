import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer

import torchvision.transforms.v2 as v2
import data.custom_transforms as custom_transforms

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from data.datamodule import BinaryImageDataModule
from models.model_transferlearning import TransferLearningModule

class OptunaTrainer:
    def __init__(self, model, config, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225], dataset_name="DwarfRabbits-binary"):
        self.model = model
        self.config = config
        self.dataset_name = dataset_name
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def _build_transform(self, image_size):
        """
        Builds a torchvision transformation pipeline for preprocessing images.

        Args:
            image_size (int): The target size (height and width) to which images will be resized.

        Returns:
            torchvision.transforms.Compose: A composed transform that applies the following steps:
                1. Center crops the image to a square using a custom transform.
                2. Resizes the image to (image_size, image_size).
                3. Converts the image to a tensor.
                4. Normalizes the tensor using ImageNet mean and standard deviation.
        """
        return v2.Compose([
            custom_transforms.CenterCropSquare(),
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            v2.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])

    def _setup_wandb_logger(self):
        """
        Sets up and returns a WandbLogger instance for experiment tracking.
        This method generates a unique experiment name based on the current configuration
        parameters and the current timestamp. It updates the configuration dictionary with
        the generated experiment name and initializes a WandbLogger with relevant
        hyperparameters and metadata for tracking the experiment in Weights & Biases.
        Returns:
            WandbLogger: An initialized WandbLogger object configured for the current experiment.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = (
            f"TL_bs{self.config['batch_size']}"
            f"_img{self.config['image_size']}"
            f"_opt{self.config['optimizer']}"
            f"_lr{self.config['learning_rate']:.0e}"
            f"_wd{self.config['weight_decay']:.0e}"
            f"_sch_{self.config['scheduler'] if self.config['scheduler'] else 'None'}"
            f"_{now_str}"
        )
        self.config['wandb_experiment_name'] = exp_name

        return WandbLogger(
            project=self.config['wandb_project_name'],
            name=exp_name,
            config={
                'sweep_id': self.config['sweep_id'],
                'batch_size': self.config['batch_size'],
                'image_size': self.config['image_size'],
                'max_epochs': self.config['max_epochs'],
                'accumulate_grad_batches': self.config['accumulate_grad_batches'],
                'precision': self.config['precision'],
                # 'optimizer': self.config['optimizer'],        # Already logged by WandbLogger
                'learning_rate': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay'],
                # 'scheduler': self.config['scheduler'],        # Already logged by WandbLogger
                'dataset': self.dataset_name,
            }
        )

    def run_training(self, trial):
        """
        Runs a single training session for hyperparameter optimization using the provided Optuna trial.
        This method suggests and sets various hyperparameters using the Optuna trial object, prepares the data module and model,
        sets up logging with Weights & Biases, and initializes the PyTorch Lightning Trainer. It then trains the model and saves
        the checkpoint. If an exception occurs during training, it logs the error and returns infinity as the loss.
        Args:
            trial (optuna.trial.Trial): The Optuna trial object used to suggest hyperparameters.
        Returns:
            float: The validation loss after training, or float("inf") if training failed.
        """
        # Suggest hyperparameters
        self.config['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128, 192])
        self.config['image_size'] = trial.suggest_categorical("image_size", [128, 192, 256])

        self.config['max_epochs'] = trial.suggest_int("max_epochs", 10, 50)
        self.config['accumulate_grad_batches'] = trial.suggest_categorical("accumulate_grad_batches", [1, 2, 4])
        self.config['precision'] = trial.suggest_categorical("precision", ["16-mixed", 32])

        self.config['optimizer'] = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
        self.config['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        self.config['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        self.config['scheduler'] = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR", None])

        # Setup Wandb Logger
        wandb_logger = self._setup_wandb_logger()

        # Prepare transformations and datamodule
        transform = self._build_transform(self.config['image_size'])
        dm = BinaryImageDataModule(
            data_dir=self.config['path_to_split_aug_pics'],
            transform=transform,
            batch_size=self.config['batch_size'],
            num_workers=2,
            persistent_workers=True
        )

        # Initialize Lightning model
        lightning_model = TransferLearningModule(
            self.model,
            learning_rate=self.config['learning_rate'],
            optimizer_name=self.config['optimizer'],
            weight_decay=self.config['weight_decay'],
            scheduler_name=self.config['scheduler'],
        )

        # Setup trainer
        trainer = Trainer(
            max_epochs=self.config['max_epochs'],
            precision=self.config['precision'],
            accumulate_grad_batches=self.config['accumulate_grad_batches'],
            accelerator="auto",
            devices="auto",
            strategy="auto",
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode='min')],
            logger=wandb_logger,
            enable_progress_bar=False,
            log_every_n_steps=10,
        )

        # Train & handle exceptions
        try:
            trainer.fit(model=lightning_model, datamodule=dm)
            checkpoint_path = f"checkpoints/{self.config['wandb_experiment_name']}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Error during training: {e}")
            wandb.finish()
            return float("inf")

        val_loss = trainer.callback_metrics.get("val_loss")
        wandb.finish()

        return val_loss.item() if val_loss else float("inf")
