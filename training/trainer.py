"""Main DGM training loop.

Implements the two-phase training protocol:
    Phase 1 (first 50% of steps): higher LR, payoff smoothing annealed
    Phase 2 (remaining steps): lower LR, exact payoff, PDE refinement
    Optional L-BFGS fine-tuning at the end.
"""

import json
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from configs.base_config import ExperimentConfig
from evaluation.diagnostics import check_nan_outputs
from evaluation.metrics import relative_l2_error
from evaluation.monte_carlo import MonteCarloPricer
from losses.combined_loss import compute_combined_loss
from pde.operator import black_scholes_operator
from pde.payoffs import get_payoff_fn, smoothed_payoff
from samplers.adaptive_sampler import AdaptiveSampler
from samplers.risk_neutral_sampler import RiskNeutralSampler
from samplers.uniform_sampler import UniformSampler
from training.callbacks import CheckpointSaver
from training.scheduler import build_scheduler
from utils.io_utils import save_config_artifact, save_json
from utils.logging import MetricsLogger, get_logger
from utils.random import seed_everything

logger = get_logger(__name__)


class DGMTrainer:
    """Main training loop for the Deep Galerkin Method.

    Parameters
    ----------
    model : nn.Module
        DGM or MLP network.
    config : ExperimentConfig
        Full experiment configuration.
    payoff_fn : callable
        Clipped payoff function (x -> Phi(x)).
    payoff_unclipped_fn : callable
        Unclipped payoff function (x -> f(x), before max).
    mc_reference : dict or None
        Pre-computed MC reference prices at test points.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        payoff_fn: Callable,
        payoff_unclipped_fn: Callable,
        mc_reference: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.model = model
        self.payoff_fn = payoff_fn
        self.payoff_unclipped_fn = payoff_unclipped_fn
        self.mc_reference = mc_reference

        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        self.model.to(self.device)

        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        m = config.market
        self.sigma_t = torch.tensor(m.sigma, dtype=torch.float32, device=self.device)
        self.rho_t = torch.tensor(m.rho, dtype=torch.float32, device=self.device)

        self._build_sampler()
        self._build_operator()
        self._build_optimizer()

        self.checkpoint = CheckpointSaver(str(self.output_dir / "checkpoints"))
        self.metrics_logger = MetricsLogger(
            str(self.output_dir), config.name, use_wandb=False, config=None,
        )

        self.loss_history: Dict[str, List[float]] = {
            "loss_pde": [], "loss_terminal": [], "loss_boundary": [], "loss_total": [],
        }
        self.eval_history: Dict[str, List[float]] = {"step": [], "rel_l2_error": []}

    def _build_sampler(self) -> None:
        m = self.config.market
        sc = self.config.sampler
        sigma_np = np.array(m.sigma)
        rho_np = np.array(m.rho)

        base = RiskNeutralSampler(
            sigma_np, rho_np, m.r, m.T, sc.domain_std_multiplier, self.device
        )

        if sc.sampler_type == "uniform":
            self.sampler = UniformSampler(sigma_np, m.T, sc.domain_std_multiplier, self.device)
        elif sc.sampler_type == "risk_neutral":
            self.sampler = base
        elif sc.sampler_type == "adaptive":
            self.sampler = AdaptiveSampler(
                base, n_candidates_multiplier=10, warmup_steps=2000, device=self.device
            )
        else:
            self.sampler = base

    def _build_operator(self) -> None:
        m = self.config.market

        def operator_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return black_scholes_operator(
                self.model, t, x, self.sigma_t, self.rho_t, m.r
            )

        self.operator_fn = operator_fn

        if isinstance(self.sampler, AdaptiveSampler):
            self.sampler.set_model_and_operator(self.model, operator_fn)

    def _build_optimizer(self) -> None:
        tc = self.config.training
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=tc.lr_init
        )
        self.scheduler = build_scheduler(
            self.optimizer, tc.scheduler, tc.T_0, tc.T_mult, tc.lr_min
        )

    def _get_eps(self, step: int) -> float:
        """Get current payoff smoothing epsilon."""
        mc = self.config.model
        tc = self.config.training
        anneal_steps = int(tc.n_steps * mc.payoff_smoothing_anneal_fraction)
        if step >= anneal_steps:
            return 0.0
        frac = step / anneal_steps
        return mc.payoff_smoothing_eps_init * (1.0 - frac)

    def _get_payoff_fn_for_step(self, step: int) -> Callable:
        """Return the (possibly smoothed) payoff function for this step."""
        eps = self._get_eps(step)
        if eps > 0.0:
            return lambda x: smoothed_payoff(self.payoff_unclipped_fn, x, eps)
        return self.payoff_fn

    def _training_step(self, step: int) -> Dict[str, torch.Tensor]:
        """Execute a single gradient step."""
        self.model.train()
        sc = self.config.sampler
        tc = self.config.training
        m = self.config.market

        t_int, x_int = self.sampler.sample_interior(sc.n_interior)
        x_term = self.sampler.sample_terminal(sc.n_terminal)
        t_bnd, x_bnd = self.sampler.sample_boundary(sc.n_boundary)

        current_payoff = self._get_payoff_fn_for_step(step)

        loss_dict = compute_combined_loss(
            self.model,
            t_int, x_int, x_term, t_bnd, x_bnd,
            self.operator_fn,
            current_payoff,
            m.T,
            lambda_pde=tc.lambda_pde,
            lambda_terminal=tc.lambda_terminal,
            lambda_boundary=tc.lambda_boundary,
            use_hard_terminal=self.config.model.use_hard_terminal_constraint,
        )

        self.optimizer.zero_grad()
        loss_dict["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), tc.grad_clip_norm
        )
        self.optimizer.step()
        self.scheduler.step()

        if isinstance(self.sampler, AdaptiveSampler):
            self.sampler.step()

        return loss_dict

    def _eval_step(self, step: int) -> Dict[str, float]:
        """Evaluate against MC ground truth."""
        self.model.eval()
        metrics = {}

        if self.mc_reference is not None:
            S0_test = self.mc_reference["S0_test"]
            mc_prices = self.mc_reference["mc_prices"]
            m = self.config.market

            x_test = np.log(S0_test / m.K)
            x_t = torch.tensor(x_test, dtype=torch.float32, device=self.device)
            t_t = torch.zeros(len(x_test), 1, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                dgm_prices = self.model(t_t, x_t).cpu().numpy().ravel()

            metrics["rel_l2_error"] = relative_l2_error(dgm_prices, mc_prices)
            metrics["max_error"] = float(np.max(np.abs(dgm_prices - mc_prices)))

        return metrics

    def _lbfgs_finetune(self) -> None:
        """L-BFGS fine-tuning phase with a fixed batch."""
        tc = self.config.training
        sc = self.config.sampler
        m = self.config.market

        logger.info("Starting L-BFGS fine-tuning (%d steps)...", tc.lbfgs_steps)

        t_int, x_int = self.sampler.sample_interior(sc.n_interior)
        x_term = self.sampler.sample_terminal(sc.n_terminal)
        t_bnd, x_bnd = self.sampler.sample_boundary(sc.n_boundary)

        t_int = t_int.detach()
        x_int = x_int.detach()

        lbfgs = torch.optim.LBFGS(
            self.model.parameters(), lr=0.1, line_search_fn="strong_wolfe",
            max_iter=20,
        )

        for lbfgs_step in range(tc.lbfgs_steps):
            def closure():
                lbfgs.zero_grad()
                t_i = t_int.requires_grad_(True)
                x_i = x_int.requires_grad_(True)
                loss_dict = compute_combined_loss(
                    self.model,
                    t_i, x_i, x_term, t_bnd, x_bnd,
                    self.operator_fn,
                    self.payoff_fn,
                    m.T,
                    lambda_pde=tc.lambda_pde,
                    lambda_terminal=tc.lambda_terminal,
                    lambda_boundary=tc.lambda_boundary,
                    use_hard_terminal=self.config.model.use_hard_terminal_constraint,
                )
                loss = loss_dict["loss_total"]
                loss.backward()
                return loss

            lbfgs.step(closure)

            if (lbfgs_step + 1) % 100 == 0:
                loss_val = closure()
                logger.info(
                    "L-BFGS step %d/%d, loss=%.6e",
                    lbfgs_step + 1, tc.lbfgs_steps, loss_val.item()
                )

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns
        -------
        dict
            Final metrics including loss history and evaluation results.
        """
        tc = self.config.training
        seed_everything(tc.seed)
        save_config_artifact(self.config, str(self.output_dir))

        if not check_nan_outputs(self.model, self.config.market.d, self.device):
            raise RuntimeError("Model produces NaN on random inputs before training!")

        logger.info(
            "Starting DGM training: %d steps, device=%s, d=%d",
            tc.n_steps, self.device, self.config.market.d,
        )
        start_time = time.time()
        start_step = 0

        latest_ckpt_path = self.output_dir / "checkpoints" / "latest_model.pt"
        if latest_ckpt_path.exists():
            try:
                ckpt = torch.load(latest_ckpt_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["state_dict"])
                start_step = ckpt.get("step", 0) + 1
                logger.info(f"Resuming training from step {start_step} using {latest_ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to load latest_model.pt: {e}")

        for step in tqdm(range(start_step, tc.n_steps), desc="DGM Training", disable=False):
            loss_dict = self._training_step(step)

            if step % tc.log_every == 0:
                metrics = {k: v.item() for k, v in loss_dict.items()}
                metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                metrics["eps"] = self._get_eps(step)
                self.metrics_logger.log(metrics, step)

                for k in self.loss_history:
                    self.loss_history[k].append(loss_dict[k].item())

            if step % tc.eval_every == 0 and step > 0:
                eval_metrics = self._eval_step(step)
                if eval_metrics:
                    self.metrics_logger.log(eval_metrics, step)
                    self.eval_history["step"].append(step)
                    self.eval_history["rel_l2_error"].append(
                        eval_metrics.get("rel_l2_error", float("nan"))
                    )
                    self.checkpoint(
                        self.model,
                        eval_metrics.get("rel_l2_error", float("inf")),
                        step,
                    )
                    logger.info(
                        "Step %d: rel_l2=%.4e, loss=%.4e",
                        step,
                        eval_metrics.get("rel_l2_error", float("nan")),
                        loss_dict["loss_total"].item(),
                    )

            if step < 100 and not check_nan_outputs(self.model, self.config.market.d, self.device):
                raise RuntimeError(f"Model produced NaN at step {step}")

        if tc.lbfgs_finetune:
            self._lbfgs_finetune()

        train_time = time.time() - start_time
        logger.info("Training complete in %.1f seconds", train_time)

        final_eval = self._eval_step(tc.n_steps)
        self.checkpoint.save_latest(self.model, tc.n_steps)

        results = {
            "train_time_seconds": train_time,
            "final_loss": self.loss_history["loss_total"][-1] if self.loss_history["loss_total"] else None,
            "loss_history": self.loss_history,
            "eval_history": self.eval_history,
            **final_eval,
        }

        save_json(results, str(self.output_dir / "results.json"))
        self.metrics_logger.close()

        return results
