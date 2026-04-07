"""Tests for payoff functions at known points."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from pde.payoffs import (
    basket_call_payoff,
    geometric_basket_payoff,
    max_call_payoff,
    spread_option_payoff,
    smoothed_payoff,
    basket_call_payoff_unclipped,
)


def test_basket_call_at_money():
    """At the money (x=0), basket call payoff is 0."""
    x = torch.zeros(1, 2)
    p = basket_call_payoff(x, K=1.0)
    assert abs(p.item()) < 1e-6, f"ATM basket call should be 0, got {p.item()}"


def test_basket_call_in_the_money():
    """Deep ITM basket call."""
    x = torch.tensor([[1.0, 1.0]])
    p = basket_call_payoff(x, K=1.0)
    expected = 1.0 * (0.5 * (torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(1.0))) - 1.0)
    assert abs(p.item() - expected.item()) < 1e-5


def test_basket_call_out_of_money():
    """OTM basket call should be 0."""
    x = torch.tensor([[-2.0, -2.0]])
    p = basket_call_payoff(x, K=1.0)
    assert abs(p.item()) < 1e-6


def test_geometric_basket():
    x = torch.tensor([[0.5, 0.5]])
    p = geometric_basket_payoff(x, K=1.0)
    expected = 1.0 * max(torch.exp(torch.tensor(0.5)).item() - 1.0, 0.0)
    assert abs(p.item() - expected) < 1e-5


def test_max_call():
    x = torch.tensor([[0.5, -0.5]])
    p = max_call_payoff(x, K=1.0)
    expected = 1.0 * max(torch.exp(torch.tensor(0.5)).item() - 1.0, 0.0)
    assert abs(p.item() - expected) < 1e-5


def test_spread():
    x = torch.tensor([[0.5, -0.5]])
    p = spread_option_payoff(x, K=1.0)
    e1 = torch.exp(torch.tensor(0.5)).item()
    e2 = torch.exp(torch.tensor(-0.5)).item()
    expected = max(e1 - e2 - 1.0, 0.0)
    assert abs(p.item() - expected) < 1e-5


def test_smoothed_payoff_converges():
    """As eps->0, smoothed payoff should converge to exact payoff."""
    x = torch.tensor([[0.3, 0.3]])
    exact = basket_call_payoff(x, K=1.0)
    for eps in [0.1, 0.01, 0.001]:
        smooth = smoothed_payoff(basket_call_payoff_unclipped, x, eps, K=1.0)
        err = abs(smooth.item() - exact.item())
        if eps <= 0.001:
            assert err < 0.01, f"Smoothed payoff not converging: eps={eps}, err={err}"


if __name__ == "__main__":
    test_basket_call_at_money()
    test_basket_call_in_the_money()
    test_basket_call_out_of_money()
    test_geometric_basket()
    test_max_call()
    test_spread()
    test_smoothed_payoff_converges()
    print("All payoff tests PASSED.")
