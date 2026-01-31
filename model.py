import torch
import torch.nn as nn
import torch.autograd as autograd

class LagrangianMLP(nn.Module):
    r"""
    A Multi-Layer Perceptron (MLP) approximating the Lagrangian scalar field :math:`\mathcal{L}(q, \dot{q})`.

    This network acts as the scalar potential function from which all physical dynamics are derived.
    To ensure the existence of the Hessian matrix (required for the Euler-Lagrange equations),
    the network employs :math:`C^\infty` smooth activation functions (Softplus).

    Args:
        input_dim (int): Dimensionality of the state space :math:`2 \times d`, where :math:`d` is the number of degrees of freedom.
        hidden_dim (int): Width of the hidden layers. Default: 64.
        num_layers (int): Number of hidden layers. Default: 3.

    Attributes:
        net (nn.Sequential): The underlying feed-forward neural network.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Softplus())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        
        # Output layer (scalar Lagrangian L)
        layers.append(nn.Linear(hidden_dim, 1, bias=False))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights (Xavier/Glorot initialization is optimal for Softplus/Tanh)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, q, q_dot):
        r"""
        Computes the Lagrangian scalar field.

        Args:
            q (torch.Tensor): Generalized coordinates of shape `(Batch, Dim)`.
            q_dot (torch.Tensor): Generalized velocities of shape `(Batch, Dim)`.

        Returns:
            torch.Tensor: The Lagrangian :math:`\mathcal{L}(q, \dot{q})` of shape `(Batch, 1)`.
        """
        # Concatenate q and q_dot along the last dimension to form state vector x
        x = torch.cat([q, q_dot], dim=-1)
        return self.net(x)

class LegendreNN(nn.Module):
    r"""
    Legendre-Transformed Neural Network (L-HNN).

    This architecture learns the Lagrangian :math:`\mathcal{L}(q, \dot{q})` via a neural network,
    but enforces the dynamics derived from the implicit Hamiltonian :math:`\mathcal{H}(q, p)` 
    constructed via the Legendre transform:
    
    .. math::
        \mathcal{H}(q, p) = p \cdot \dot{q} - \mathcal{L}(q, \dot{q})

    where :math:`p = \nabla_{\dot{q}}\mathcal{L}` is the canonical momentum.

    The forward pass solves the continuous-time Euler-Lagrange equations by inverting the Mass Matrix (Hessian):

    .. math::
        M(q, \dot{q}) \ddot{q} + C(q, \dot{q}) \dot{q} = \nabla_q \mathcal{L}

    where :math:`M = \nabla^2_{\dot{q}} \mathcal{L}` and :math:`C = \nabla^2_{q, \dot{q}} \mathcal{L}`.

    Args:
        input_dim (int): Number of degrees of freedom (d). Total input dimension is :math:`2d`.
        hidden_dim (int, optional): Hidden layer width. Default: 64.
        num_layers (int, optional): Number of hidden layers. Default: 3.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.dim = input_dim
        # The MLP models the Lagrangian L
        self.lagrangian_net = LagrangianMLP(input_dim * 2, hidden_dim, num_layers)

    def compute_lagrangian(self, q, q_dot):
        r"""
        Wrapper to compute the scalar Lagrangian :math:`\mathcal{L}`.
        """
        return self.lagrangian_net(q, q_dot)

    def forward(self, q, q_dot):
        r"""
        Predicts the generalized acceleration :math:`\ddot{q}` given the current state :math:`(q, \dot{q})`.

        The method solves the linear system :math:`M \cdot \ddot{q} = \text{Forces}` using a differentiable
        linear solver with Tikhonov regularization on the diagonal of :math:`M` to ensure positive definiteness.

        Args:
            q (torch.Tensor): Generalized coordinates `(Batch, Dim)`.
            q_dot (torch.Tensor): Generalized velocities `(Batch, Dim)`.
            
        Returns:
            torch.Tensor: Predicted generalized acceleration :math:`\ddot{q}` of shape `(Batch, Dim)`.
        """
        # Enable gradient computation for higher-order derivatives (Hessians)
        # essential even during inference (torch.no_grad context)
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            q_dot = q_dot.detach().requires_grad_(True)
            
            # 1. Compute Lagrangian L
            # Shape: (Batch, 1)
            L = self.compute_lagrangian(q, q_dot)
            
            # 2. Compute gradients (First Derivatives)
            # Use autograd.grad with create_graph=True to allow for second derivatives later.
            
            # Canonical Momentum: p = dL/dq_dot
            # Shape: (Batch, Dim)
            p = autograd.grad(L.sum(), q_dot, create_graph=True)[0]
            
            # Generalized Force Partial: dL/dq
            # Shape: (Batch, Dim)
            dL_dq = autograd.grad(L.sum(), q, create_graph=True)[0]
            
            # 3. Compute Hessians (Second Derivatives)
            # M = d^2L / (dq_dot^2)      (Mass Matrix)
            # C = d^2L / (dq_dot * dq)   (Mixed Partial / Coriolis-like terms)
            
            batch_size = q.size(0)
            dim = self.dim
            
            M = torch.zeros(batch_size, dim, dim, device=q.device)
            C = torch.zeros(batch_size, dim, dim, device=q.device) 
            
            # Iterate over degrees of freedom to construct the batch Hessian matrices
            for i in range(dim):
                # Gradients of the i-th component of momentum w.r.t q_dot
                grad_p_i = autograd.grad(p[:, i].sum(), q_dot, create_graph=True)[0]
                M[:, i, :] = grad_p_i
                
                # Gradients of the i-th component of momentum w.r.t q
                grad_p_i_q = autograd.grad(p[:, i].sum(), q, create_graph=True)[0]
                C[:, i, :] = grad_p_i_q
            
            # 4. Solve the Euler-Lagrange System
            # Equation: M * q_ddot = dL_dq - (d(dL/dq_dot)/dt without q_ddot term)
            # Expansion: M * q_ddot + C * q_dot = dL_dq
            # Rearranged: M * q_ddot = dL_dq - C @ q_dot
            
            q_dot_unsqueezed = q_dot.unsqueeze(-1)  # (Batch, Dim, 1)
            
            # Term: C @ q_dot
            # Shape: (Batch, Dim, 1)
            term2 = torch.bmm(C, q_dot_unsqueezed).squeeze(-1)
            
            # Right-Hand Side (Generalized Forces)
            RHS = dL_dq - term2
            
            # Regularize Mass Matrix M to ensure invertibility
            # M_reg = M + epsilon * I
            epsilon = 1e-6
            eye = torch.eye(dim, device=q.device).unsqueeze(0).expand(batch_size, -1, -1)
            M_reg = M + epsilon * eye
            
            # Solve Linear System: M_reg * q_ddot = RHS
            # torch.linalg.solve is more stable than .inverse()
            q_ddot = torch.linalg.solve(M_reg, RHS.unsqueeze(-1)).squeeze(-1)
            
        return q_ddot

    def get_hamiltonian(self, q, q_dot):
        r"""
        Computes the implicit Hamiltonian :math:`\mathcal{H}` from the learned Lagrangian.

        .. math::
            \mathcal{H} = p \cdot \dot{q} - \mathcal{L}

        Args:
            q (torch.Tensor): Coordinates.
            q_dot (torch.Tensor): Velocities.

        Returns:
            torch.Tensor: The scalar Energy/Hamiltonian `(Batch, 1)`.
        """
        L = self.compute_lagrangian(q, q_dot)
        p = autograd.grad(L.sum(), q_dot, create_graph=True)[0]
        # H = dot(p, q_dot) - L
        H = (p * q_dot).sum(dim=1, keepdim=True) - L
        return H