import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from utils.math_utils import validate_correlation_matrix
import numpy as np
from configs.base_config import MarketConfig
from models.fbsde_network import DeepBSDENetwork

class DeepBSDETrainer:
    """Trainer for the Deep BSDE (Han, Jentzen, E 2018) solver."""
    
    def __init__(
        self,
        market: MarketConfig,
        model: DeepBSDENetwork,
        payoff_fn,
        n_steps: int = 100000,
        batch_size: int = 4096,
        num_time_steps: int = 100,
        lr: float = 1e-2,
        device: torch.device = torch.device('cpu')
    ):
        self.market = market
        self.model = model.to(device)
        self.payoff_fn = payoff_fn
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_time_steps = num_time_steps
        self.device = device
        
        self.dt = market.T / num_time_steps
        
        # Optimizer and scheduler based on Han et al. 2018
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[n_steps//2, (3*n_steps)//4], gamma=0.1)
        
        # Precompute constants
        self.d = market.d
        self.r = market.r
        sigma = torch.tensor(market.sigma, device=device)
        rho = torch.tensor(market.rho, device=device)
        
        # Drift in log coordinates: mu = r - 0.5 * sigma^2
        self.mu = self.r - 0.5 * sigma**2
        self.sigma = sigma
        
        # Cholesky factor for diffusion
        L_np = validate_correlation_matrix(np.array(market.rho))
        self.L = torch.tensor(L_np, dtype=torch.float32, device=device)
        
        # Fixed initial log-price X_0
        self.X_0 = torch.log(torch.tensor(market.S0, device=device) / market.K)

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Initialize paths
        X = self.X_0.expand(self.batch_size, self.d)
        Y = self.model.y_init.expand(self.batch_size, 1)
        
        # Euler-Maruyama loop
        for step in range(self.num_time_steps):
            Z = self.model(X, step)
            
            # Brownian increments dW ~ N(0, dt)
            dW = torch.randn(self.batch_size, self.d, device=self.device) * (self.dt ** 0.5)
            
            # Forward SDE for log-price X: dX = mu dt + diag(sigma) L dW
            X = X + self.mu * self.dt + torch.matmul(dW, self.L.T) * self.sigma
            
            # Backward SDE for value Y: dY = r Y dt + Z dW
            # Z is shape (B, d), dW is shape (B, d). Dot product per sample:
            Z_dW = torch.sum(Z * dW, dim=1, keepdim=True)
            Y = Y + self.r * Y * self.dt + Z_dW
            
        # Terminal condition
        payoff = self.payoff_fn(X).view(-1, 1)
        
        loss = torch.nn.functional.mse_loss(Y, payoff)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()

    def train(self):
        print(f"Training Deep BSDE for {self.n_steps} steps...")
        start_time = time.time()
        
        pbar = tqdm(range(self.n_steps), desc="Deep BSDE Training")
        for step in pbar:
            loss = self.train_step()
            
            if step % 100 == 0:
                y0_val = self.model.y_init.item()
                pbar.set_postfix({"Loss": f"{loss:.4e}", "Y_0": f"{y0_val:.6f}"})
                
        time_taken = time.time() - start_time
        print(f"Training complete in {time_taken:.1f}s. Final Y_0 = {self.model.y_init.item():.6f}")
        return self.model
