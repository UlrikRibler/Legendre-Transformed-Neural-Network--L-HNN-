# Legendre-Transformed Neural Network (L-HNN)

**A Unified Deep Learning Framework for Lagrangian and Hamiltonian Mechanics**

## Abstract

This repository implements the **Legendre-Transformed Neural Network (LegendreNN)**, a rigorous physics-informed architecture that bridges the gap between Lagrangian Neural Networks (LNN) and Hamiltonian Neural Networks (HNN). 

While LNNs offer data efficiency by operating in the observable configuration space $(q, \dot{q})$, they often lack the long-term stability provided by the symplectic structure of HNNs. Conversely, HNNs require canonical coordinates $(q, p)$, which are rarely directly observable in real-world systems (e.g., robotics).

**LegendreNN** solves this dichotomy by:
1.  Approximating the scalar Lagrangian field $\mathcal{L}(q, \dot{q})$ via a neural network.
2.  Implicitly constructing the Hamiltonian $\mathcal{H}(q, p)$ via the Legendre transform.
3.  Deriving dynamics that strictly adhere to symplectic conservation laws while consuming standard state observations.

## Mathematical Formulation

### The Unified Manifold
The network learns a parameterized Lagrangian $\mathcal{L}_\theta(q, \dot{q})$. The forward dynamics are governed by the Euler-Lagrange equations, solved via a differentiable linear system involving the Hessian (Mass Matrix):

$$ M(q, \dot{q}) \ddot{q} + C(q, \dot{q}) \dot{q} = \nabla_q \mathcal{L} $$

Where:
*   $M = \nabla^2_{\dot{q}} \mathcal{L}$ is the generalized Mass Matrix.
*   $C = \nabla^2_{q, \dot{q}} \mathcal{L}$ represents Coriolis and centrifugal terms.

### The Legendre Transform
Unlike standard black-box models, the internal state representation is lifted to the cotangent bundle (phase space) via the fiber derivative:

$$ p = \frac{\partial \mathcal{L}}{\partial \dot{q}} $$

The system's total energy (Hamiltonian) is thus constructed as:

$$ \mathcal{H}(q, p) = \langle p, \dot{q} \rangle - \mathcal{L}(q, \dot{q}) $$

This ensures that the learned dynamics $\ddot{q}$ are conservative and reversible by construction.

## Architecture

The implementation leverages PyTorch's automatic differentiation engine to compute higher-order derivatives on the fly.

*   **Manifold Smoothness:** Uses `Softplus` activations ($\beta=1$) to ensuring $C^\infty$ continuity, guaranteeing the existence of the Hessian $M$.
*   **Robust Solver:** Solves the linear system $M \cdot \ddot{q} = F$ using Tikhonov regularization ($M_{reg} = M + \epsilon I$) to prevent singularities near critical points in the configuration space.
*   **Differentiable Physics:** The entire pipeline, including the linear solve, is differentiable, allowing end-to-end training against acceleration data $\ddot{q}_{gt}$.

## Superiority Scenarios

This architecture is rigorously superior in specific scientific domains:

1.  **Robotics & Control (Observable Coordinates):**
    *   *Constraint:* Sensors provide joint angles $q$ and velocities $\dot{q}$. Momentum $p$ is unknown.
    *   *Advantage:* LegendreNN accepts $(q, \dot{q})$ directly, eliminating the need for inverse dynamics pre-processing required by HNNs.

2.  **Long-Horizon Simulation:**
    *   *Constraint:* Numerical integration of learned ODEs leads to energy drift.
    *   *Advantage:* By enforcing the symplectic structure via the implicit Hamiltonian, the system is confined to the correct energy level set, preventing "exploding" or "vanishing" physical behavior over thousands of timesteps.

3.  **Variable Inertia Systems:**
    *   *Constraint:* Systems like multi-link manipulators have configuration-dependent mass matrices $M(q)$.
    *   *Advantage:* The Hessian-based formulation naturally captures these non-linear inertial effects, whereas simple MLPs fail to generalize across the state space.

## Installation

### Prerequisites
*   Python 3.8+
*   PyTorch 1.9+
*   NumPy, SciPy, Matplotlib

### Setup
```bash
git clone https://github.com/ulrikribler/LegendreNN.git
cd LegendreNN
pip install -r requirements.txt
```

## Experimentation

To reproduce the results on the canonical **Simple Pendulum** system:

```bash
python train.py
```

### Protocol
1.  **Data Generation:** 2000 phase-space samples $(q, \dot{q})$ are sampled uniformly. Ground truth accelerations are computed via analytical mechanics.
2.  **Training:** The network minimizes the $L_2$ loss on predicted acceleration.
3.  **Evaluation:** The learned vector field is integrated using `scipy.integrate.odeint` (LSODA) and compared against the ground truth trajectory.
4.  **Verification:** Conservation of the implicit Hamiltonian $\mathcal{H}$ is verified over the rollout.

### Results
The training script produces high-fidelity phase portraits and energy conservation plots, saved locally as:
*   `legendre_nn_results.png` (Trajectory Analysis)
*   `energy_check.png` (Symplectic Verification)

## Citation

If you use this code in your research, please cite the associated methodology:

```bibtex
@software{legendre_nn_2026,
  author = {Ulrik Ribler},
  title = {Legendre-Transformed Neural Network: A PyTorch Implementation},
  year = {2026},
  url = {https://github.com/ulrikribler/LegendreNN}
}
```

## License

MIT License. See `LICENSE` for details.
