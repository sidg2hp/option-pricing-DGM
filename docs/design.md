# design.md — DGM Architecture and Paper Replication

Technical design document for the DGM option pricing codebase. Covers the neural architecture, PDE formulation, and the connection to the original papers.

---

## 1. Deep Galerkin Method: Overview

The Deep Galerkin Method (Sirignano & Spiliopoulos, 2018) is a mesh-free approach for solving high-dimensional PDEs using neural networks. Unlike finite difference or finite element methods, DGM does not require a spatial mesh — it samples collocation points randomly from the domain and minimises the PDE residual in a Monte Carlo sense.

**Core idea**: Parameterise the solution $u(t, \mathbf{x})$ by a neural network $u_\theta$ and minimise:

$$\mathcal{J}(\theta) = \mathbb{E}_{(t,\mathbf{x}) \sim \mu_\Omega}\left[|\mathcal{L}[u_\theta](t,\mathbf{x})|^2\right] + \lambda_T\,\mathbb{E}_{\mathbf{x} \sim \mu_T}\left[|u_\theta(T,\mathbf{x}) - \Phi(\mathbf{x})|^2\right] + \lambda_B\,\mathbb{E}_{(t,\mathbf{x}) \sim \mu_B}\left[|B[u_\theta](t,\mathbf{x})|^2\right]$$

where $\mathcal{L}$ is the differential operator, $\Phi$ the terminal condition, and $B$ the boundary operator. All expectations are approximated by mini-batch averages over collocation points.

**Mesh-free advantage**: The method's computational cost scales with the network size, not the spatial resolution to the power $d$. This makes it tractable for $d \gg 3$ where grid-based methods are infeasible.

---

## 2. DGM Network Architecture

### 2.1 Design Motivation

Standard MLPs suffer from the _vanishing information_ problem: the original input $(t, \mathbf{x})$ is progressively transformed by each layer, making it difficult for deep networks to maintain gradient information with respect to the input coordinates. This is particularly problematic for PDE solvers, which require accurate $\partial u / \partial t$, $\partial u / \partial x_i$, and $\partial^2 u / \partial x_i \partial x_j$ via automatic differentiation.

The DGM architecture addresses this by **re-injecting the original input at every hidden layer** through an LSTM-inspired gating mechanism:

```
Input [t, x] ──┬──────────────────────────────────────────── ... ──┐
               │                                                    │
               ▼                                                    ▼
        ┌─────────────┐     ┌─────────────┐             ┌─────────────┐
        │ Initial      │     │ DGM Layer 1  │     ...     │ DGM Layer L  │
        │ Embedding    │────▶│ (gating)     │────▶ ... ──▶│ (gating)     │─── Linear ──▶ û(t,x)
        │ W₀[t,x]+b₀  │     │              │             │              │
        │ + activation │     │ Z, G, R, H   │             │ Z, G, R, H   │
        └─────────────┘     └─────────────┘             └─────────────┘
```

### 2.2 DGM Layer Equations

Each DGM layer receives two inputs: the original $(t, \mathbf{x})$ vector and the hidden state $S^l$ from the previous layer. It produces an updated hidden state $S^{l+1}$ via four gates:

| Gate | Equation | Role |
|------|----------|------|
| **Z** (carry) | $Z^l = \sigma(U^z [t,\mathbf{x}] + W^z S^l + b^z)$ | Controls how much of the previous state to retain |
| **G** (update) | $G^l = \sigma(U^g [t,\mathbf{x}] + W^g S^l + b^g)$ | Controls the mix between new candidate and carry |
| **R** (reset) | $R^l = \sigma(U^r [t,\mathbf{x}] + W^r S^l + b^r)$ | Selectively resets parts of the hidden state before computing candidate |
| **H** (candidate) | $H^l = \sigma(U^h [t,\mathbf{x}] + W^h (S^l \odot R^l) + b^h)$ | New candidate hidden state |

**State update**:

$$S^{l+1} = (1 - G^l) \odot H^l + Z^l \odot S^l$$

**Key property**: Each gate has a direct linear path from the input $[t, \mathbf{x}]$ through $U^*$. This means $\partial S^{l+1} / \partial \mathbf{x}$ has a non-vanishing component at every depth, ensuring that the PDE residual computed by autograd remains well-conditioned.

### 2.3 Weight Matrices

Each DGM layer has 4 pairs of weight matrices:

| Matrix | Dimensions | Connected to |
|--------|-----------|--------------|
| $U^z, U^g, U^r, U^h$ | $(1+d) \times h$ | Input $[t, \mathbf{x}]$ |
| $W^z, W^g, W^r, W^h$ | $h \times h$ | Previous hidden state $S^l$ |

The $W^*$ matrices have no bias (bias is absorbed into $U^*$). Total parameters per DGM layer: $4(1+d)h + 4h^2 + 4h$ (from biases in $U^*$).

### 2.4 Comparison with Standard MLP

This codebase also implements an MLP baseline (`models/mlp_network.py`) with two-layer residual blocks:

```
Block: x → Linear → σ → Linear → σ → (+x)
```

The MLP does **not** re-inject the input at each layer. The MLP baseline uses the same hard terminal constraint and training protocol, isolating the effect of the DGM gating architecture.

---

## 3. Hard Terminal Constraint Ansatz

### 3.1 Formulation

Instead of penalising deviations from the terminal condition in the loss (soft constraint), we structurally enforce:

$$u_\theta(t, \mathbf{x}) = \underbrace{\Phi(\mathbf{x})\,e^{-r(T-t)}}_{\text{discounted payoff}} + \underbrace{(T-t)}_{\to 0 \text{ at } T}\,\hat{u}_\theta(t, \mathbf{x})$$

**Verification**: At $t = T$: $(T - T) = 0$, so $u_\theta(T, \mathbf{x}) = \Phi(\mathbf{x}) \cdot e^{0} = \Phi(\mathbf{x})$. ✓

### 3.2 Advantages

1. **Exact enforcement**: No training needed for the terminal condition; it holds regardless of $\theta$.
2. **Reduced loss landscape complexity**: One fewer loss component to balance (eliminates terminal loss weight tuning).
3. **Better initialisation**: At $t = 0$ with $\hat{u}_\theta \approx 0$ (Xavier init), the network output approximates $\Phi(\mathbf{x})\,e^{-rT}$, which is the discounted payoff — already a reasonable first-order approximation.

### 3.3 The Non-Differentiability Problem

The payoff $\Phi(\mathbf{x}) = \max(f(\mathbf{x}), 0)$ has a kink at $f = 0$. The second derivative $\frac{\partial^2}{\partial x^2}\max(f(x), 0)$ contains a Dirac delta at the kink.

Since the hard constraint embeds $\Phi$ in the network output, PyTorch's autograd computes $\frac{\partial^2 u_\theta}{\partial x_i \partial x_j}$ including the non-differentiable payoff term. `torch.clamp` returns zero gradients at the kink, producing **incorrect** PDE residuals.

### 3.4 Smoothed Payoff Fix

During training, replace $\max(f, 0)$ with:

$$\Phi_\epsilon(\mathbf{x}) = \frac{1}{2}\left(f(\mathbf{x}) + \sqrt{f(\mathbf{x})^2 + \epsilon^2}\right)$$

Properties:
- $\Phi_\epsilon \to \max(f, 0)$ as $\epsilon \to 0$
- $\Phi_\epsilon$ is $C^\infty$ for all $\epsilon > 0$
- $\Phi_\epsilon'(f) = \frac{1}{2}\left(1 + \frac{f}{\sqrt{f^2 + \epsilon^2}}\right)$ — smooth sigmoid-like transition
- $\Phi_\epsilon(0) = \epsilon/2$ (non-zero bias that vanishes as $\epsilon \to 0$)

**Annealing schedule**: $\epsilon(s) = \epsilon_0 \cdot (1 - s/s_{\text{anneal}})$ for $s < s_{\text{anneal}}$, then $\epsilon = 0$. Default: $\epsilon_0 = 0.1$, $s_{\text{anneal}} = 0.3 \cdot n_{\text{steps}}$.

At evaluation and for inference, the exact (clipped) payoff is always used.

---

## 4. PDE Operator Implementation

### 4.1 Operator Structure

The Black-Scholes operator in log-price coordinates:

$$\mathcal{L}[u] = \frac{\partial u}{\partial t} + \frac{1}{2}\sum_{i,j}A_{ij}\frac{\partial^2 u}{\partial x_i \partial x_j} + \sum_i \mu_i \frac{\partial u}{\partial x_i} - r\,u$$

where $A_{ij} = \rho_{ij}\sigma_i\sigma_j$ and $\mu_i = r - \sigma_i^2 / 2$.

### 4.2 Automatic Differentiation Pipeline

1. Forward pass: $u = u_\theta(t, \mathbf{x})$
2. First derivatives: `torch.autograd.grad(u, t)` → $\partial u / \partial t$ and `torch.autograd.grad(u, x)` → $\nabla_x u$, both with `create_graph=True`
3. Second derivatives: For each $i$, `torch.autograd.grad(du_dx[:, i].sum(), x)` → row $i$ of the Hessian, with `create_graph=True` (needed for backpropagation through the residual)

### 4.3 Hutchinson's Trace Estimator ($d > 5$)

For $d > 5$, the exact Hessian requires $d$ backward passes. Instead, use the stochastic identity:

$$\text{tr}(AH) = \mathbb{E}_{\mathbf{v} \sim \text{Rad}^d}\left[\mathbf{v}^\top A H \mathbf{v}\right]$$

where $\mathbf{v} \in \{-1, +1\}^d$ is a Rademacher vector. The Hessian-vector product $H\mathbf{v}$ is computed with a single backward pass:

```python
Hv = torch.autograd.grad((du_dx * v).sum(), x, create_graph=True)[0]
```

This reduces cost from $O(d)$ to $O(m)$ backward passes, where $m = 20$ is the number of Rademacher samples. The variance is $O(1/m)$.

---

## 5. Sampling Strategy

### 5.1 Risk-Neutral Sampler (Default)

Interior points: $t \sim \text{Uniform}[0, T]$, $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_x, \Sigma_x)$ where:

- Mean: $\mu_{x,i} = (r - \sigma_i^2/2) \cdot T/2$ (expected log-price at mid-maturity)
- Covariance: $\Sigma_{x,ij} = \rho_{ij}\sigma_i\sigma_j \cdot T$

Terminal enrichment: 20% of terminal samples drawn from $\mathcal{N}(0, 0.1^2 I)$ to enrich the at-the-money region where the payoff kink causes the highest PDE residuals.

Boundary samples: One randomly chosen coordinate is set to $x_i^{\min} = -k\sigma_i\sqrt{T}$ (lower domain boundary).

### 5.2 Adaptive Sampler

After a warmup period (2000 steps), the adaptive sampler:

1. Draws $10n$ candidate points from the risk-neutral sampler
2. Evaluates $|\mathcal{L}[u_\theta]|^2$ at each candidate (no gradient)
3. Resamples $n$ points proportional to the squared residual (importance sampling)

This concentrates collocation points in regions where the current solution is least accurate.

---

## 6. Zhou et al. (2021) Replication

### 6.1 Method Description

Zhou et al. propose a _regression_ approach: train a feedforward NN $f_\theta(\mathbf{S}_0) \approx V(0, \mathbf{S}_0)$ using MC-generated training data.

**Algorithm**:
1. Sample $N$ random initial conditions $\mathbf{S}_{0,k}$ from a log-uniform distribution: $\mathbf{S}_{0,k} = K \exp(\text{Uniform}(-0.5, 0.5))^d$
2. For each $\mathbf{S}_{0,k}$, average 50 MC paths to estimate $V_k = \mathbb{E}[e^{-rT}\Phi(\mathbf{S}_T) | \mathbf{S}_0 = \mathbf{S}_{0,k}]$
3. Train NN via MSE: $\min_\theta \frac{1}{N}\sum_k |f_\theta(\mathbf{S}_{0,k}) - V_k|^2$

**Architecture**: 4 hidden layers, 100 neurons each, ReLU activation.

**Key limitation**: The NN only learns $V(0, \mathbf{S}_0)$ — it cannot evaluate the price at arbitrary $(t, \mathbf{x})$ and does not produce Greeks as by-products.

### 6.2 Differences from DGM

| Aspect | DGM | Zhou et al. |
|--------|-----|-------------|
| What it solves | Full PDE on $(t, \mathbf{x}) \in [0,T] \times \mathbb{R}^d$ | Price at $t=0$ only |
| Training data | None (PDE residual is the loss) | MC simulation |
| MC paths required | 0 during training | $50 \times N$ ($N$ = training samples) |
| Greeks | Free (autograd) | Requires finite differencing |
| MC noise | No label noise | Label noise from averaged MC paths |
| Evaluation | Full price surface + time evolution | Pointwise at $t=0$ |

### 6.3 Replication Results (This Work)

Using $d=2$, $K=100$, the DGM approach produces the full price surface while Zhou's method provides point estimates only. At the 5 test points, both methods underestimate OTM prices relative to 1M-path MC, with DGM showing tighter convergence at ITM strikes. See the main README for the detailed comparison table.

---

## 7. Loss Function Design

The full loss functional:

$$\mathcal{J}(\theta) = \lambda_\mathcal{L}\,\underbrace{\frac{1}{N_\Omega}\sum_{k=1}^{N_\Omega}|\mathcal{L}[u_\theta](t_k, \mathbf{x}_k)|^2}_{\text{PDE residual}} + \lambda_\mathcal{T}\,\underbrace{\frac{1}{N_T}\sum_{k=1}^{N_T}|u_\theta(T, \mathbf{x}_k) - \Phi(\mathbf{x}_k)|^2}_{\text{terminal (soft only)}} + \lambda_\mathcal{B}\,\underbrace{\frac{1}{N_B}\sum_{k=1}^{N_B}|u_\theta(t_k, \mathbf{x}_k)|^2}_{\text{boundary}}$$

With the hard terminal constraint, $\lambda_\mathcal{T} = 0$ (terminal condition is structural).

**Boundary condition**: For call-type payoffs, $V(\mathbf{S}) \to 0$ as any $S_i \to 0$. In log-price coordinates, this corresponds to $u(t, \mathbf{x}) \to 0$ as $x_i \to -\infty$. The boundary loss penalises $|u_\theta|^2$ at points where one coordinate is at the lower domain boundary.

---

## 8. Training Implementation Details

### 8.1 Two-Phase Training

**Phase 1** (steps 0 to $0.3 \cdot n_{\text{steps}}$): Smoothed payoff with linearly decaying $\epsilon$. Higher effective learning rates from cosine warm restarts.

**Phase 2** (remaining steps): Exact payoff ($\epsilon = 0$). Lower learning rate from cosine annealing. The network refines the PDE approximation with the true non-smooth payoff.

### 8.2 L-BFGS Fine-tuning

After Adam training, optionally run L-BFGS with strong Wolfe line search on a fixed batch of collocation points. This performs quasi-Newton optimisation, which converges faster near a minimum. The L-BFGS phase typically reduces the PDE residual by 1–2 orders of magnitude.

### 8.3 Cosine Annealing with Warm Restarts

The learning rate follows the schedule of Loshchilov & Hutter (2017):

$$\eta(s) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{s \bmod T_i}{T_i}\pi\right)\right)$$

where $T_i = T_0 \cdot T_{\text{mult}}^{i}$ is the period of cycle $i$. The warm restarts help escape shallow local minima in the PDE loss landscape.

---

## 9. Monte Carlo Reference Pricer

The MC pricer uses exact simulation (no Euler discretisation):

$$S_i(T) = S_i(0)\,\exp\!\left((r - \sigma_i^2/2)\,T + \sigma_i\sqrt{T}\,Z_i\right)$$

where $\mathbf{Z} = L\boldsymbol{\varepsilon}$, $L$ is the Cholesky factor of $\rho$, and $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$.

**Variance reduction**: Antithetic variates — pair each $\boldsymbol{\varepsilon}$ with $-\boldsymbol{\varepsilon}$ to create negatively correlated path pairs, reducing variance by exploiting the symmetry of the normal distribution.

---

## 10. Greeks via Automatic Differentiation

Since $u_\theta$ is a differentiable neural network, Greeks are computed exactly via autograd:

**Delta** (sensitivity to asset price):

$$\Delta_i = \frac{\partial V}{\partial S_i} = \frac{1}{S_i}\frac{\partial u}{\partial x_i} = \frac{1}{K e^{x_i}}\frac{\partial u}{\partial x_i}$$

**Gamma** (convexity):

$$\Gamma_i = \frac{\partial^2 V}{\partial S_i^2} = \frac{1}{S_i^2}\left(\frac{\partial^2 u}{\partial x_i^2} - \frac{\partial u}{\partial x_i}\right)$$

The chain rule from log-price to price coordinates introduces the $1/S_i$ and $1/S_i^2$ factors and the subtraction in gamma (from the log transformation's second derivative).

**Accuracy**: For 1D Black-Scholes, the delta error is 0.22% and gamma error is 2.95%. Gamma's higher error is inherent — it involves second derivatives of the network, where approximation errors are amplified.

---

## 11. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING STEP                                 │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Sampler   │───▶│ (t_int,x_int)│    │ (x_terminal) │               │
│  │           │    │ (t_bnd,x_bnd)│    │              │               │
│  └──────────┘    └──────┬───────┘    └──────┬───────┘               │
│                         │                    │                        │
│                         ▼                    ▼                        │
│  ┌───────────────────────────────────────────────────────┐           │
│  │                  u_θ (DGM Network)                    │           │
│  │  [t,x] → Initial Layer → DGM₁ → DGM₂ → ... → DGMₗ → Linear   │
│  │         ↑_________________↑_______↑___________↑                  │
│  │         (input re-injection at every layer)                       │
│  │                                                                   │
│  │  Hard constraint: u = Φ(x)·exp(-rτ) + τ·û                       │
│  └──────────┬────────────────────────┬──────────────────┘           │
│             │                        │                               │
│             ▼                        ▼                               │
│  ┌───────────────────┐    ┌──────────────────────┐                  │
│  │ PDE Operator L[u] │    │ Boundary: |u|²       │                  │
│  │ = ∂u/∂t + ½Σ Aᵢⱼ │    │ at x_min faces       │                  │
│  │   ∂²u/∂xᵢ∂xⱼ +  │    └──────────┬───────────┘                  │
│  │   Σ μᵢ ∂u/∂xᵢ   │               │                               │
│  │   - ru            │               │                               │
│  └──────────┬────────┘               │                               │
│             │                        │                               │
│             ▼                        ▼                               │
│  ┌───────────────────────────────────────────────────────┐           │
│  │  J(θ) = λ_L · mean(|L[u]|²) + λ_B · mean(|u_bnd|²) │           │
│  └──────────────────────────┬────────────────────────────┘           │
│                             │                                        │
│                             ▼                                        │
│                    ┌─────────────────┐                               │
│                    │ Adam / L-BFGS   │                               │
│                    │ ∇_θ J(θ)        │                               │
│                    └─────────────────┘                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Log-price coordinates | Constant PDE coefficients; better numerical conditioning for NN |
| Tanh activation (not ReLU) | ReLU has zero second derivative → destroys PDE residual |
| Hard terminal constraint | Exact enforcement eliminates a loss component and improves stability |
| Smoothed payoff annealing | Non-differentiable kink in `max` breaks autograd Hessian computation |
| Risk-neutral sampling | Concentrates collocation points where the solution has the most variation |
| Cosine warm restarts | Periodic LR spikes help escape local minima in the non-convex loss |
| L-BFGS fine-tuning | Second-order method converges faster near a minimum; polishes PDE residual |
| Hutchinson estimator ($d > 5$) | Avoids $O(d)$ backward passes; cost becomes $O(m)$ independent of $d$ |
| Xavier initialisation | Maintains variance across layers; standard for tanh networks |
| Gradient clipping (norm 1.0) | Prevents PDE loss spikes from destabilising training |
