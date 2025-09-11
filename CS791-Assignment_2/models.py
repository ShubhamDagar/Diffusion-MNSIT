import torch.nn as nn

class D3PM(nn.Module):
    def __init__(self): # Add any required parameters
        super().__init__()
        # Define your model architecture here

class ConditionalD3PM(nn.Module):
    def __init__(self, num_classes): # Add any required parameters
        super().__init__()
        self.num_classes = num_classes
        # Define your conditional model architecture here

class DDPM(nn.Module):
    def __init__(self): # Add any required parameters
        super().__init__()
        # Define your model architecture here

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes): # Add any required parameters
        super().__init__()
        self.num_classes = num_classes
        # Define your conditional model architecture here