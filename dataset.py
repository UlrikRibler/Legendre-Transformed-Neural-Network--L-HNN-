import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import odeint

class PendulumDataset(Dataset):
    r"""
    Synthetic dataset generator for a Simple Pendulum system.
    
    The system dynamics are governed by the Hamiltonian:
    
    .. math::
        \mathcal{H} = \frac{p_\theta^2}{2ml^2} + mgl(1 - \cos\theta)
        
    Resulting in the equation of motion (acceleration):
    
    .. math::
        \ddot{\theta} = -\frac{g}{l} \sin\theta
        
    Args:
        samples (int): Number of phase space samples to generate.
        t_span (float): Time horizon for integration (unused for random sampling).
        noise_std (float): Standard deviation of Gaussian noise added to inputs.
    """
    def __init__(self, samples=1000, t_span=10.0, noise_std=0.0):
        self.samples = samples
        self.noise_std = noise_std
        
        # Physical Parameters
        self.g = 9.81  # Gravity (m/s^2)
        self.l = 1.0   # Length (m)
        self.m = 1.0   # Mass (kg)
        
        # Generate phase space samples
        self.data = self._generate_data(samples, t_span)
        
    def _generate_data(self, samples, t_span):
        """
        Generates random samples uniformly distributed in the relevant phase space.
        Computes ground-truth acceleration using analytical mechanics.
        """
        # Phase Space Sampling:
        # q (Angle) ~ U[-pi, pi]
        # q_dot (Angular Velocity) ~ U[-5, 5]
        q = np.random.uniform(-np.pi, np.pi, (samples, 1))
        q_dot = np.random.uniform(-5, 5, (samples, 1))
        
        # Analytical Equation of Motion:
        # q_ddot = -(g/l) * sin(q)
        q_ddot = -(self.g / self.l) * np.sin(q)
        
        # Add Measurement Noise (if applicable)
        if self.noise_std > 0:
            q += np.random.normal(0, self.noise_std, q.shape)
            q_dot += np.random.normal(0, self.noise_std, q_dot.shape)
        
        # Cast to float32 for PyTorch training
        X = np.concatenate([q, q_dot], axis=1).astype(np.float32)
        Y = q_ddot.astype(np.float32)
        
        return X, Y

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Returns:
            q (torch.Tensor): Angle [rad]
            q_dot (torch.Tensor): Angular Velocity [rad/s]
            q_ddot (torch.Tensor): Ground Truth Angular Acceleration [rad/s^2]
        """
        state = self.data[0][idx]
        target = self.data[1][idx]
        
        q = torch.tensor(state[:1]) 
        q_dot = torch.tensor(state[1:])
        q_ddot = torch.tensor(target)
        
        return q, q_dot, q_ddot

def get_pendulum_trajectory(t_span=10, steps=1000, initial_state=[np.pi/4, 0]):
    """
    Integrates the true pendulum dynamics using LSODA (scipy.integrate.odeint).
    Used for ground-truth comparison during evaluation.
    """
    t = np.linspace(0, t_span, steps)
    g = 9.81
    l = 1.0
    
    def func(state, t):
        theta, theta_dot = state
        return [theta_dot, -(g/l) * np.sin(theta)]
    
    trajectory = odeint(func, initial_state, t)
    return t, trajectory