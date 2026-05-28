import torch
import numpy as np

from configs.base_config import MarketConfig
from models.fbsde_network import DeepBSDENetwork
from training.fbsde_trainer import DeepBSDETrainer
from pde.payoffs import get_payoff_fn

class DeepBSDEPricer:
    """Wrapper class for Deep BSDE to fit into the comparison framework.
    
    Since Deep BSDE is a point-wise solver, it is trained specifically for a single
    initial state X_0 (usually the At-The-Money point S_0 = K).
    """
    
    def __init__(self, market: MarketConfig, n_steps: int = 20000, device: torch.device = torch.device('cpu')):
        self.market = market
        self.n_steps = n_steps
        self.device = device
        
        # We only train it for the ATM point (where S_0 = K)
        # So we ensure market.S0 is precisely K
        self.market.S0 = [market.K] * market.d
        
        self.payoff_fn, _ = get_payoff_fn(self.market.payoff_type, self.market.K)
        self.model = DeepBSDENetwork(d=self.market.d, num_time_steps=50, hidden_size=64)
        
        self.trainer = DeepBSDETrainer(
            market=self.market,
            model=self.model,
            payoff_fn=self.payoff_fn,
            n_steps=self.n_steps,
            batch_size=1024,
            num_time_steps=50,
            lr=1e-2,
            device=self.device
        )
        
    def train(self):
        """Train the Deep BSDE model for the specified ATM point."""
        self.trainer.train()
        
    def price(self, S0_grid: np.ndarray) -> np.ndarray:
        """Return the prices for the requested S0 points.
        
        Since this is a pointwise solver, it only returns the valid price if the
        requested S0 matches the ATM point it was trained on. Otherwise it returns NaN.
        """
        M = S0_grid.shape[0]
        prices = np.full(M, np.nan)
        
        # Find which point in the grid is ATM (S0 = K)
        atm_val = self.market.K
        for i in range(M):
            if np.allclose(S0_grid[i], atm_val):
                # The model's trained parameter y_init is the price at X_0
                prices[i] = self.model.y_init.item()
                
        return prices
