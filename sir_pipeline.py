"""
SIR Model Simulation Pipeline

This script reads daily infection data from a CSV file, fits SIR model parameters
(β and ν) to the data, and exports simulation results for Excel plotting.

Author: Generated for MAT238
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


def simulate_sir(beta: float, nu: float, S0: int, I0: int, R0: int, N: int, days: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates SIR model using Euler's Method.
    
    SIR Model Equations:
        dS/dt = -beta * S * I / N
        dI/dt = beta * S * I / N - nu * I
        dR/dt = nu * I
    
    Euler's Method:
        For each time step t:
            S(t+1) = S(t) + h * dS/dt
            I(t+1) = I(t) + h * dI/dt
            R(t+1) = R(t) + h * dR/dt
        where h = 1 (1 day step size)
    
    Args:
        beta: Infection rate (transmission rate)
        nu: Recovery rate
        S0: Initial susceptible population
        I0: Initial infected population
        R0: Initial recovered population
        N: Total population (S + I + R)
        days: Number of days to simulate
    
    Returns:
        Tuple of (S, I, R) arrays, each of length (days + 1)
    """
    # Initialize arrays to store S, I, R values
    S = np.zeros(days + 1)
    I = np.zeros(days + 1)
    R = np.zeros(days + 1)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Step size for Euler's Method (1 day)
    h = 1.0
    
    # Euler's Method iteration
    for t in range(days):
        # Calculate derivatives at current time step
        dS_dt = -beta * S[t] * I[t] / N
        dI_dt = beta * S[t] * I[t] / N - nu * I[t]
        dR_dt = nu * I[t]
        
        # Update values using Euler's Method
        S[t + 1] = S[t] + h * dS_dt
        I[t + 1] = I[t] + h * dI_dt
        R[t + 1] = R[t] + h * dR_dt
        
        # Ensure non-negative values (population cannot be negative)
        S[t + 1] = max(0, S[t + 1])
        I[t + 1] = max(0, I[t + 1])
        R[t + 1] = max(0, R[t + 1])
    
    return S, I, R


def objective_function(params: np.ndarray, data: pd.DataFrame, S0: int, I0: int, R0: int, N: int) -> float:
    """
    Objective function for parameter fitting.
    Calculates mean squared error between simulated and observed infection data.
    
    Args:
        params: Array [beta, nu] to be optimized
        data: DataFrame with 'date' and 'infected' columns
        S0, I0, R0: Initial conditions
        N: Total population
    
    Returns:
        Mean squared error (MSE) between simulated and observed infections
    """
    beta, nu = params
    
    # Number of days to simulate (based on data length)
    days = len(data)
    
    # Run simulation
    S, I, R = simulate_sir(beta, nu, S0, I0, R0, N, days)
    
    # Calculate mean squared error between simulated I and observed infections
    observed_I = data['infected'].values
    mse = np.mean((I[:len(observed_I)] - observed_I) ** 2)
    
    return mse


def fit_parameters(data: pd.DataFrame, S0: int, I0: int, R0: int, N: int) -> Tuple[float, float]:
    """
    Fits SIR model parameters β and ν to minimize MSE between simulated and observed data.
    
    Uses scipy.optimize.minimize with L-BFGS-B method, with fallback to
    differential_evolution for global optimization if needed.
    
    Args:
        data: DataFrame with 'date' and 'infected' columns
        S0: Initial susceptible population
        I0: Initial infected population
        R0: Initial recovered population
        N: Total population
    
    Returns:
        Tuple of (optimal_beta, optimal_nu)
    """
    # Initial parameter guesses
    # beta: typically between 0.1 and 1.0 for daily transmission
    # nu: typically between 0.05 and 0.5 for recovery rate (1/recovery_time)
    initial_beta = 0.3
    initial_nu = 0.1
    
    # Parameter bounds: [beta_min, beta_max], [nu_min, nu_max]
    bounds = [(0.001, 2.0), (0.001, 1.0)]
    
    # Try local optimization first (faster)
    try:
        result = minimize(
            objective_function,
            x0=[initial_beta, initial_nu],
            args=(data, S0, I0, R0, N),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            beta_opt, nu_opt = result.x
            print(f"Optimization successful (L-BFGS-B): beta = {beta_opt:.6f}, nu = {nu_opt:.6f}")
            return beta_opt, nu_opt
    except Exception as e:
        print(f"Local optimization failed: {e}")
        print("Trying global optimization (differential_evolution)...")
    
    # Fallback to global optimization
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(data, S0, I0, R0, N),
        seed=42,
        maxiter=1000,
        popsize=15
    )
    
    beta_opt, nu_opt = result.x
    print(f"Optimization successful (differential_evolution): beta = {beta_opt:.6f}, nu = {nu_opt:.6f}")
    return beta_opt, nu_opt


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads infection data from CSV file.
    
    Expected format:
        date, infected
    
    Args:
        csv_path: Path to input CSV file
    
    Returns:
        DataFrame with 'date' and 'infected' columns
    """
    data = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'date' not in data.columns or 'infected' not in data.columns:
        raise ValueError("CSV must contain 'date' and 'infected' columns")
    
    # Convert date column to datetime if it's not already
    data['date'] = pd.to_datetime(data['date'])
    
    # Ensure infected values are numeric
    data['infected'] = pd.to_numeric(data['infected'], errors='coerce')
    
    # Remove any rows with missing values
    data = data.dropna()
    
    return data


def export_results(data: pd.DataFrame, S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                   beta: float, nu: float, output_path: str):
    """
    Exports simulation results to CSV file for Excel plotting.
    
    Columns: date, S, I, R, observed_I, beta, nu
    
    Args:
        data: Original data DataFrame with dates
        S, I, R: Simulated arrays from SIR model
        beta, nu: Fitted parameters
        output_path: Path to output CSV file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare export DataFrame
    export_data = pd.DataFrame({
        'date': data['date'].values,
        'S': S[:len(data)],
        'I': I[:len(data)],
        'R': R[:len(data)],
        'observed_I': data['infected'].values,
        'beta': beta,
        'nu': nu
    })
    
    # Export to CSV
    export_data.to_csv(output_path, index=False)
    print(f"Results exported to: {output_path}")


def plot_results(data: pd.DataFrame, S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                 beta: float, nu: float, save_path: str = None):
    """
    Plots simulated vs observed infection curves.
    
    Args:
        data: Original data DataFrame
        S, I, R: Simulated arrays
        beta, nu: Fitted parameters
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    days = len(data)
    dates = data['date'].values
    
    # Plot 1: Infections (Simulated vs Observed)
    axes[0].plot(dates, I[:days], 'b-', label='Simulated I(t)', linewidth=2)
    axes[0].scatter(dates, data['infected'].values, color='red', marker='o', 
                    label='Observed Infections', s=50, alpha=0.7)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Number of Infected')
    axes[0].set_title(f'SIR Model Fit: Infections (beta = {beta:.4f}, nu = {nu:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: All compartments (S, I, R)
    axes[1].plot(dates, S[:days], 'g-', label='Susceptible (S)', linewidth=2)
    axes[1].plot(dates, I[:days], 'b-', label='Infected (I)', linewidth=2)
    axes[1].plot(dates, R[:days], 'r-', label='Recovered (R)', linewidth=2)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Population')
    axes[1].set_title('SIR Model: All Compartments')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='SIR Model Simulation Pipeline - Fit parameters and export results for Excel'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/cases.csv',
        help='Path to input CSV file with date and infected columns (default: data/cases.csv)'
    )
    parser.add_argument(
        '--population',
        type=int,
        default=1000000,
        help='Total population N (default: 1000000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/sir_simulation.csv',
        help='Path to output CSV file (default: output/sir_simulation.csv)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Display plot of simulation results'
    )
    parser.add_argument(
        '--plot-save',
        type=str,
        default=None,
        help='Path to save plot image (optional)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = load_data(args.data)
    print(f"Loaded {len(data)} data points")
    
    # Set initial conditions
    # I0: first observed infection count
    I0 = int(data['infected'].iloc[0])
    # R0: assume no recovered initially
    R0 = 0
    # S0: remaining population
    S0 = args.population - I0 - R0
    
    print(f"\nInitial Conditions:")
    print(f"  Total Population (N): {args.population:,}")
    print(f"  Initial Susceptible (S0): {S0:,}")
    print(f"  Initial Infected (I0): {I0:,}")
    print(f"  Initial Recovered (R0): {R0:,}")
    
    # Fit parameters
    print(f"\nFitting parameters beta and nu...")
    beta, nu = fit_parameters(data, S0, I0, R0, args.population)
    
    # Run simulation
    print(f"\nRunning SIR simulation...")
    days = len(data)
    S, I, R = simulate_sir(beta, nu, S0, I0, R0, args.population, days)
    
    # Export results
    print(f"\nExporting results...")
    export_results(data, S, I, R, beta, nu, args.output)
    
    # Optional plotting
    if args.plot or args.plot_save:
        print(f"\nGenerating plot...")
        plot_results(data, S, I, R, beta, nu, args.plot_save)
    
    print(f"\nPipeline completed successfully!")


if __name__ == '__main__':
    main()

