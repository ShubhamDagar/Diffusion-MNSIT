import torch
import math

class NoiseSchedulerDDPM():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)

        self.alphas = torch.cumprod(1 - self.betas, dim=0)
    
    def init_cosine_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        # taken from https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf 

        s = 0.008
        T = self.num_timesteps

        self.alphas = torch.zeros(T, dtype=torch.float32)
        for t in range(T):
            arg_t = (((t / (T - 1)) + s) / (1 + s)) * math.pi / 2   # FIX: T-1
            arg_0 = (s / (1 + s)) * math.pi / 2
            f_t = math.cos(arg_t) ** 2
            f_0 = math.cos(arg_0) ** 2
            self.alphas[t] = f_t / f_0

        # Compute betas safely
        self.betas = torch.zeros(T, dtype=torch.float32)
        self.betas[0] = min(1 - self.alphas[0], 0.999)
        for t in range(1, T):
            self.betas[t] = min(1 - self.alphas[t] / self.alphas[t-1], 0.999)
    
        
    def __len__(self):
        return self.num_timesteps
    
class MaskSchedulerD3PM():
    """
    Mask scheduler for Discrete Diffusion (D3PM) models.

    Args:
        num_timesteps: int, number of timesteps in the diffusion process
        mask_type: str, type of mask scheduling ("uniform", "linear", etc.)
        **kwargs: additional arguments for mask scheduling

    This object sets up the mask schedule for each timestep.
    """

    def __init__(self, num_timesteps=50, mask_type="uniform", **kwargs):
        self.num_timesteps = num_timesteps
        self.mask_type = mask_type

        if mask_type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

    def init_linear_schedule(self):
        """
        Initializes a linear mask schedule where the mask probability increases linearly.
        """
        self.mask_probs = None

    def __len__(self):
        return self.num_timesteps
