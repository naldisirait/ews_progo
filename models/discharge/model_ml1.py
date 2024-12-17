import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegression(nn.Module):
    def __init__(self, input_size=72, output_size=72, hidden_size=128, num_hidden_layers=4, dropout_rate=0.2):
        """
        Initializes an advanced MLP regression model.

        Parameters:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_size (int): Number of neurons in each hidden layer.
            num_hidden_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(MLPRegression, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass of the MLP model.
        """
        # Input layer with activation
        x = F.relu(self.input_layer(x))
        
        # Hidden layers with dropout, layer normalization, and activation
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return x

def load_model_f24(device,path_trained_f24):
    input_size = 72
    output_size = 72
    hidden_size = 128
    num_hidden_layers = 4
    dropout_rate = 0.2

    # Initialize the model
    model_f24 = MLPRegression(
        input_size=input_size, 
        output_size=output_size, 
        hidden_size=hidden_size, 
        num_hidden_layers=num_hidden_layers, 
        dropout_rate=dropout_rate)
    
    # Load the state_dict (weights) into the model
    model_f24.load_state_dict(torch.load(path_trained_f24,map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode if you are using it for inference
    model_f24.to(device)
    model_f24.eval()
    
    return model_f24

def inference_ml1(precip, config):
    """
    Function to predict discharge
    Args:
        precip(tensor): grided precipitation with shape (Batch=1, len_history=72, width=8, height=7)
    Returns:
        discharge(tensor): 72 hours of discharge, where 48 hours is estimated and 24 hours forcast discharge.
    """
    #set constants
    device = "cpu"
    precip = precip.float() #make sure the value type is float
    length_discharge_to_extract = 72
    
    #load model
    path_trained_f24 = config['model']['path_trained_ml1_f24']
    model_f24 = load_model_f24(device,path_trained_f24=path_trained_f24)

    #inference model with given input precipitation
    with torch.no_grad():
        output_f13 = model_f24(precip)
        B,T = output_f24.shape
        output_f24 = output_f24.view(B,T,1) # Add a new dimension, making the tensor of shape (B, T, feature)
    discharge = output_f24.view(-1)[-length_discharge_to_extract:]
    
    return discharge