# model.py

# required imports
import torch
from torch import nn
import pytorch_lightning as pl

class block_template(nn.Module):

    """
    A PyTorch module that defines a reusable block template for 2D convolutional layers.
    This block consists of a 2D convolutional layer, followed by a ReLU activation function
    and a Batch Normalization layer. The convolutional layer uses 'same' padding to ensure
    the output size matches the input size (if stride is 1).
    Attributes:
        net (nn.Sequential): A sequential container holding the convolutional layer, 
            ReLU activation, and batch normalization.
    Args:
        in_channels (int): Number of input channels for the convolutional layer.
        out_channels (int): Number of output channels for the convolutional layer.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolutional operation.
    Methods:
        forward(x):
            Defines the forward pass of the block. Takes an input tensor `x` and
            applies the sequential layers to it.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(block_template, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same', bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.net(x)
    
class Model_Template(pl.LightningModule):

    """
    Model_Template is a PyTorch Lightning module designed for building and training a neural network model.
    Attributes:
        in_channels (int): Number of input channels for the model.
        layer_n (int): Number of layers or size of the input sequence.
        out_channels (int): Number of output channels for the model. Default is 1.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        loss (nn.Module): Loss function used for training the model.
        example_input_array (torch.Tensor): Example input tensor for model tracing.
        padding (int): Padding size calculated based on the kernel size.
        AvgPool1D1 (nn.Module): Average pooling layer with kernel size 2 and stride 2.
        layer1 (nn.Sequential): Sequential container for the first convolutional layer.
    Methods:
        forward(x):
            Defines the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after forward pass.
        configure_optimizers():
            Configures the optimizer for training.
            Returns:
                torch.optim.Optimizer: Optimizer instance.
        training_step(batch, batch_idx):
            Defines the training step for a single batch.
            Args:
                batch (tuple): A tuple containing input data and target labels.
                batch_idx (int): Index of the batch.
            Returns:
                torch.Tensor: Computed loss for the batch.
        test_step(batch, batch_idx):
            Defines the test step for a single batch.
            Args:
                batch (tuple): A tuple containing input data and target labels.
                batch_idx (int): Index of the batch.
            Returns:
                torch.Tensor: Computed loss for the batch.
        validation_step(batch, batch_idx):
            Defines the validation step for a single batch.
            Args:
                batch (tuple): A tuple containing input data and target labels.
                batch_idx (int): Index of the batch.
            Returns:
                torch.Tensor: Computed loss for the batch.
        on_test_epoch_end():
            Callback executed at the end of the test epoch.
    """

    def __init__(self, in_channels, layer_n, out_channels=1, kernel_size=3):
        super(Model_Template, self).__init__()

        # Allow to save hyperparameters
        self.save_hyperparameters()

        # TODO: Define used loss function here
        # self.loss = nn.BCEWithLogitsLoss() # 20250214_04
        # self.loss = nn.CrossEntropyLoss() # 20250214_03
        self.loss = nn.BCELoss() # 20250214_05 # 20250215_01 # 20250215_02 # 20250221_01
    
        self.example_input_array = torch.rand(1, in_channels, layer_n)

        self.in_channels = in_channels
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Calculate padding and convert padding to int
        self.padding = int(((self.kernel_size - 1) / 2))

        # Define pooling operations
        # TODO: Define pooling operations here
        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Apply 2 1d-convolutional layers
        # TODO: Define layers for model here
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64 * factor, kernel_size=kernel_size, stride=1, padding=self.padding)
        )
    
    # TODO: Implement a forward pass
    def forward(self, x):

        # Debugging print statements
        enablePrint = False
        if enablePrint: print(x.size())

        # Define the forward pass
        x_hat = 1

        # Return prediction
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        print('Test Epoch End')
        print('-----------------------------------')