import torch
import torch.nn as nn

class SubNetwork(nn.Module):
    """A simple MLP for approximating Z_t at a single time step."""
    def __init__(self, d: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, d)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepBSDENetwork(nn.Module):
    """Deep BSDE Network (Han, Jentzen, E 2018).
    
    Predicts the initial price Y_0 and the gradient Z_t at each time step.
    This is a point-wise solver: Y_0 and Z_0 are learned parameters for a fixed initial state X_0.
    """
    def __init__(self, d: int, num_time_steps: int, hidden_size: int = 256):
        super().__init__()
        self.d = d
        self.num_time_steps = num_time_steps
        
        # Learnable parameters for the fixed initial state X_0
        self.y_init = nn.Parameter(torch.zeros(1))
        self.z_init = nn.Parameter(torch.zeros(1, d))
        
        # N-1 sub-networks for t_1, ..., t_{N-1}
        self.sub_networks = nn.ModuleList([
            SubNetwork(d, hidden_size) for _ in range(num_time_steps - 1)
        ])
        
    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Evaluate Z_t at time step `step`.
        
        Parameters
        ----------
        x : torch.Tensor
            State at time t_step. Shape (batch_size, d).
        step : int
            The current time step index (0 to num_time_steps - 1).
            
        Returns
        -------
        torch.Tensor
            The gradient Z_t. Shape (batch_size, d).
        """
        if step == 0:
            # At t=0, x is identical for all paths (X_0), so Z_0 is a single parameter
            batch_size = x.shape[0]
            return self.z_init.expand(batch_size, -1)
        else:
            # At t > 0, we use the corresponding sub-network
            return self.sub_networks[step - 1](x)
