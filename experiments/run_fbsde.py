import argparse
import torch
import numpy as np

from configs.base_config import MarketConfig
from models.fbsde_network import DeepBSDENetwork
from training.fbsde_trainer import DeepBSDETrainer
from pde.payoffs import get_payoff_fn
from evaluation.monte_carlo import MonteCarloPricer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=20000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    d = args.d
    
    market = MarketConfig(
        d=d,
        r=0.05,
        T=1.0,
        K=1.0,
        sigma=[0.2] * d,
        rho=[[1.0 if i==j else 0.3 for j in range(d)] for i in range(d)],
        S0=[1.0] * d,
        payoff_type='geometric_basket' if d > 1 else 'basket_call'
    )
    
    payoff_fn, _ = get_payoff_fn(market.payoff_type, market.K)
    
    model = DeepBSDENetwork(d=d, num_time_steps=50, hidden_size=64)
    trainer = DeepBSDETrainer(
        market=market,
        model=model,
        payoff_fn=payoff_fn,
        n_steps=args.n_steps,
        batch_size=1024,
        num_time_steps=50,
        lr=1e-2,
        device=device
    )
    
    trainer.train()
    
    y0 = model.y_init.item()
    print(f"Deep BSDE Price at S0=1.0: {y0:.6f}")
    
    # Calculate MC reference for S0=1.0
    print("Calculating MC reference...")
    pricer = MonteCarloPricer()
    # MonteCarloPricer uses a numpy function for payoff, but our payoff_fn is for torch tensors!
    # Wait! MonteCarloPricer uses payoff_fn which expects numpy arrays!
    # Our get_payoff_fn returns a torch tensor function.
    # We should define a simple numpy wrapper.
    def np_payoff(S_T):
        t_S = torch.tensor(np.log(S_T/market.K), dtype=torch.float32)
        return payoff_fn(t_S).numpy().ravel()
        
    mc_result = pricer.price(
        S0=np.array(market.S0),
        payoff_fn=np_payoff,
        r=market.r,
        sigma=np.array(market.sigma),
        rho=np.array(market.rho),
        T=market.T,
        n_paths=500000
    )
    mc_price = mc_result['price']
    print(f"MC Reference Price: {mc_price:.6f}")
    
    rel_error = abs(y0 - mc_price) / mc_price * 100
    print(f"Relative Error: {rel_error:.2f}%")

if __name__ == '__main__':
    main()
