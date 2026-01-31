import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint

from model import LegendreNN
from dataset import PendulumDataset, get_pendulum_trajectory

def train():
    """
    Main training routine for the Legendre-Transformed Neural Network.
    
    Procedure:
    1. Instantiates the LegendreNN model and the Pendulum environment.
    2. Optimizes the Lagrangian parameters to minimize the L2 error between 
       predicted acceleration and ground-truth acceleration (derived from analytical physics).
    3. Evaluates the learned dynamics by performing a long-horizon integration.
    """
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 500         
    LEARNING_RATE = 1e-3
    INPUT_DIM = 1         # 1 DOF (Degree of Freedom) for Simple Pendulum
    
    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # --- Data Preparation ---
    print("Initializing Pendulum Dataset...")
    train_dataset = PendulumDataset(samples=2000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Model Initialization ---
    model = LegendreNN(input_dim=INPUT_DIM).to(device)
    
    # --- Optimization Setup ---
    # Adam optimizer is generally robust for non-convex energy landscapes.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss: MSE(predicted_acceleration, true_acceleration)
    # Minimizing this loss forces the learned Lagrangian to satisfy the Euler-Lagrange equations.
    criterion = nn.MSELoss()
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for q, q_dot, q_ddot_true in train_loader:
            q = q.to(device)
            q_dot = q_dot.to(device)
            q_ddot_true = q_ddot_true.to(device)
            
            # Forward Pass:
            # Predict acceleration using implicit Lagrangian dynamics
            q_ddot_pred = model(q, q_dot)
            
            # Compute Loss
            loss = criterion(q_ddot_pred, q_ddot_true)
            
            # Backward Pass & Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Mean Loss: {avg_loss:.6e}")
            
    print("Training complete.")
    
    # Save checkpoint
    torch.save(model.state_dict(), "legendre_nn_pendulum.pth")
    print("Model saved to legendre_nn_pendulum.pth")
    
    # --- Evaluation ---
    evaluate_and_plot(model, device)

def evaluate_and_plot(model, device):
    """
    Evaluates the trained model by integrating the learned vector field.
    
    Compares:
    1. Ground Truth Trajectory (via analytical ODE).
    2. Neural Trajectory (via learned Lagrangian dynamics).
    3. Hamiltonian Evolution (Energy conservation check).
    """
    model.eval()
    
    # Simulation Parameters
    t_eval = np.linspace(0, 10, 200) # 10 seconds simulation
    y0 = [np.pi/2, 0.0]              # Initial State: 90 degrees, rest
    
    # 1. Ground Truth Integration
    g, l = 9.81, 1.0
    def true_derivs(y, t):
        theta, theta_dot = y
        return [theta_dot, -(g/l) * np.sin(theta)]
        
    print("Integrating Ground Truth dynamics...")
    traj_true = odeint(true_derivs, y0, t_eval)
    
    # 2. Learned Dynamics Integration
    # Wrapper function compatible with scipy.integrate.odeint
    def learned_derivs(y, t):
        theta, theta_dot = y
        
        # Prepare inputs (Requires Shape: 1x1)
        q = torch.tensor([[theta]], dtype=torch.float32).to(device)
        q_dot = torch.tensor([[theta_dot]], dtype=torch.float32).to(device)
        
        # Predict acceleration from the neural manifold
        # Note: model.forward() handles enable_grad internally
        q_ddot = model(q, q_dot).cpu().item()
            
        return [theta_dot, q_ddot]
    
    print("Integrating Learned Neural dynamics...")
    traj_pred = odeint(learned_derivs, y0, t_eval)
    
    # 3. Visualization
    plt.figure(figsize=(14, 6))
    
    # Plot A: Time Series (Angle vs Time)
    plt.subplot(1, 2, 1)
    plt.plot(t_eval, traj_true[:, 0], 'k--', label=r'Ground Truth ($	heta_{GT}$)', linewidth=2)
    plt.plot(t_eval, traj_pred[:, 0], 'r-', label=r'LegendreNN ($	heta_{Pred}$)', linewidth=1.5, alpha=0.9)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (rad)', fontsize=12)
    plt.title('Trajectory Rollout (t=0 to 10s)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot B: Phase Portrait (Velocity vs Position)
    plt.subplot(1, 2, 2)
    plt.plot(traj_true[:, 0], traj_true[:, 1], 'k--', label='Ground Truth Orbit', linewidth=2)
    plt.plot(traj_pred[:, 0], traj_pred[:, 1], 'r-', label='Learned Orbit', linewidth=1.5, alpha=0.9)
    plt.xlabel(r'Position $q$ (rad)', fontsize=12)
    plt.ylabel(r'Velocity $\dot{q}$ (rad/s)', fontsize=12)
    plt.title('Phase Space Topology', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('legendre_nn_results.png', dpi=300)
    print("Trajectory plots saved to legendre_nn_results.png")
    
    # 4. Hamiltonian Analysis (Energy Conservation)
    qs = torch.tensor(traj_pred[:, 0:1], dtype=torch.float32).to(device)
    q_dots = torch.tensor(traj_pred[:, 1:2], dtype=torch.float32).to(device)
    
    # Enable grad for Hamiltonian computation (p = dL/dq_dot)
    qs.requires_grad_(True)
    q_dots.requires_grad_(True)
    
    H_pred = model.get_hamiltonian(qs, q_dots).detach().cpu().numpy()
    
    # Analytical Energy
    E_true = 0.5 * traj_pred[:, 1]**2 + 9.81 * (1 - np.cos(traj_pred[:, 0]))
    
    plt.figure(figsize=(8, 5))
    plt.plot(t_eval, H_pred, 'b-', label=r'Learned Hamiltonian $\mathcal{H}$', linewidth=2)
    plt.plot(t_eval, E_true, 'k--', label=r'Analytical Energy $E$', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Energy (J)', fontsize=12)
    plt.title('Symplectic Conservation Check', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_check.png', dpi=300)
    print("Energy analysis saved to energy_check.png")

if __name__ == "__main__":
    train()