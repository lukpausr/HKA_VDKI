# %%
import os
import math
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.model_summary import ModelSummary
from torchvision import transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from data.cats_and_dogs import BinaryCIFARDataModule
from models.model_cnn_mika import CatsDogsModel
from models.model_cnn_mika import KaninchenModel
from data.cats_and_dogs import KaninchenDataModule
from data.Kaninchen_Module import BinaryImageDataModule

# %% [markdown]
# ### Loading Configuration
# 
# In the following steps, we will load the configuration settings using the `load_configuration` function. The configuration is stored in the `config` variable which will be used throughout the script.

# %%
from config.load_configuration import load_configuration
def center_crop_square(img):
        min_side = min(img.width, img.height)
        top = max(0, (img.height - min_side) // 2)
        left = max(0, (img.width - min_side) // 2)
        return transforms.functional.crop(img, top=top, left=left, height=min_side, width=min_side)

def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    config = load_configuration()

    # %% [markdown]
    # ### Logging in to Weights & Biases (wandb)
    # 
    # Before starting any experiment tracking, ensure you are logged in to your Weights & Biases (wandb) account. This enables automatic logging of metrics, model checkpoints, and experiment configurations. The following code logs you in to wandb:
    # 
    # ```python
    # wandb.login()
    # ```
    # If you are running this for the first time, you may be prompted to enter your API key.

    # %%
    # Initialize the Wandb logger
    wandb.login()

    # %% [markdown]
    # ### Setting Seeds for Reproducibility
    # 
    # To ensure comparable and reproducible results, we set the random seed using the `seed_everything` function from PyTorch Lightning. This helps in achieving consistent behavior across multiple runs of the notebook.

    # %%
    pl.seed_everything(config['seed'])
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN optimizations for reproducibility

    # %% [markdown]
    # ### Checking for GPU Devices
    # 
    # In this step, we check for the availability of GPU devices and print the device currently being used by PyTorch. This ensures that the computations are performed on the most efficient hardware available.

    # %%
    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Torch Version: ', torch.__version__)
    print('Using device: ', device)
    if device.type == 'cuda':
        print('Cuda Version: ', torch.version.cuda)
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        torch.set_float32_matmul_precision('high')

    # %% [markdown]
    # ### Defining Transformations and Instantiating DataModule
    # 
    # In this step, we will define the necessary data transformations and initialize the `Animal_DataModule` with the provided configuration.

    # %%
    # TODO: Define transformations here
    size = config['image_size']

    transform = transforms.Compose([
        transforms.Lambda(center_crop_square),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #dm = BinaryCIFARDataModule(transform=transform, batch_size=config['batch_size'], num_workers=2, persistent_workers=True)

    # must set workers to 0, otherwise it will not work with the DataLoader

    dm = BinaryImageDataModule(data_dir=config['path_to_split_aug_pics'],transform=transform, batch_size=config['batch_size'], num_workers=2, persistent_workers=True)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    print('Train dataset size:', len(dm.train_dataset))
    print('Validation dataset size:', len(dm.val_dataset))
    print('Test dataset size:', len(dm.test_dataset))


    # %% [markdown]
    # ### Creating the Model
    # 
    # In this step, we will define the model architecture and print its summary using the `ModelSummary` utility from PyTorch Lightning. This provides an overview of the model's layers, parameters, and structure.

    # %%
    #model = CatsDogsModel()
    model = KaninchenModel()
    print(ModelSummary(model, max_depth=-1))  

    # %% [markdown]
    # ### Training the Model and Logging with Weights & Biases
    # 
    # In this step, we initialize the Wandb logger and configure the experiment name to include a timestamp for better tracking. The `Trainer` from PyTorch Lightning is set up with the Wandb logger and an early stopping callback to monitor validation loss and prevent overfitting. After training, the Wandb run is finished, and the trained model checkpoint is saved with a unique filename containing the current date and time.

    # %%
    # Initialize the Wandb logger
    # add time to the name of the experiment
    import datetime
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config['wandb_project_name'],
        name=config['wandb_experiment_name'] + '_' + current_time,
        config={
            #'dataset': 'CIFAR-binary',
            'dataset': 'Kaninchen',
            'batch_size': config['batch_size'],
            'max_epochs': config['max_epochs'],
            'learning_rate': config['learning_rate']
        }
    )

    # Initialize Trainer with wandb logger, using early stopping callback (https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
    trainer = Trainer(
        max_epochs=config['max_epochs'], 
        default_root_dir='model/checkpoint/', #data_directory, 
        accelerator="auto", 
        devices="auto", 
        strategy="auto",
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')], 
        logger=wandb_logger)

    # Training of the model
    trainer.fit(model=model, datamodule=dm)

    # Finish wandb
    wandb.finish()

    # Create a filename with date identifier
    model_filename = f"{config['wandb_experiment_name']}_{current_time}.ckpt"

    # Save the model's state_dict to the path specified in config
    save_path = os.path.join(os.path.dirname(config['path_to_models']), model_filename)
    trainer.save_checkpoint(save_path)
    print(f"Model checkpoint saved as {save_path}")
    config['path_to_model'] = save_path

    # %% [markdown]
    # # Predict with the Model
    # 

    # %%
    # from PIL import Image
    # import torch
    # # Load the saved model weights from the path specified in config

    # def predict_image(path, model):
    #     transform = transforms.Compose([
    #         transforms.Resize((150, 150)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5]*3, [0.5]*3)
    #     ])

    #     img = Image.open(path).convert('RGB')
    #     img = transform(img).unsqueeze(0)  # Add batch dimension

    #     model.eval()
    #     with torch.no_grad():
    #         pred = model(img)
    #         result = "Dog" if pred.item() > 0.5 else "Cat"
    #     print(f"Prediction: {result}")


    # %% [markdown]
    # ### Loading and Evaluating the Trained Model
    # 
    # The trained model is loaded from the checkpoint specified in the configuration. If the checkpoint exists, the model weights are restored and the model is set to evaluation mode. PyTorch Lightning's `Trainer` is then used to evaluate the model on the test dataset, providing a streamlined way to assess model performance after training.

    # %%
    model_path = config['path_to_model']
    if model_path and os.path.exists(model_path):
        #model = CatsDogsModel.load_from_checkpoint(model_path, map_location=device)
        model = KaninchenModel.load_from_checkpoint(model_path, map_location=device)
        print(f"Loaded model weights from {model_path}")
    else:
        print("Model path not found or not specified in config.")

    # Ensure model is in eval mode
    model.eval()

    # Pytorch Lightning's Trainer can be used to test the model
    trainer = Trainer()
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()