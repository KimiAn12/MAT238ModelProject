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


def simulate_sir(beta: float, nu: float, S0: float, I0: float, R0: float, N: float,
                 steps: int, time_step: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates SIR model using Euler's Method with configurable time step.
    
    SIR Model Equations:
        dS/dt = -beta * S * I / N
        dI/dt = beta * S * I / N - nu * I
        dR/dt = nu * I
    
    Euler's Method:
        For each time step t:
            S(t+1) = S(t) + h * dS/dt
            I(t+1) = I(t) + h * dI/dt
            R(t+1) = R(t) + h * dR/dt
        where h = time_step (in days)
    
    Args:
        beta: Infection rate (transmission rate)
        nu: Recovery rate
        S0: Initial susceptible population
        I0: Initial infected population
        R0: Initial recovered population
        N: Total population (S + I + R)
        steps: Number of discrete steps (usually len(data))
        time_step: Duration (in days) represented by each row
   
    Returns:
        Tuple of (S, I, R) arrays, each of length (steps + 1)
    """
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    
    # Initialize arrays to store S, I, R values
    S = np.zeros(steps + 1)
    I = np.zeros(steps + 1)
    R = np.zeros(steps + 1)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Step size for Euler's Method (time_step days)
    h = float(time_step)
    
    # Euler's Method iteration
    for t in range(steps):
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


def objective_function(params: np.ndarray, data: pd.DataFrame, S0: float, I0: float, R0: float,
                       N: float, time_step: float, fit_target: str) -> float:
    """
    Objective function for parameter fitting.
    Calculates mean squared error between simulated and observed infection data.
    
    Args:
        params: Array [beta, nu] to be optimized
        data: DataFrame with 'date' and 'infected' columns
        S0, I0, R0: Initial conditions
        N: Total population
        time_step: Duration (days) represented by each time step
        fit_target: 'active' to match I(t), 'new_cases' to match incidence
        time_step: Duration (days) represented by each time step
    
    Returns:
        Mean squared error (MSE) between simulated and observed infections
    """
    beta, nu = params
    
    # Number of steps to simulate (based on data length)
    steps = len(data)
    
    # Run simulation
    S, I, R = simulate_sir(beta, nu, S0, I0, R0, N, steps, time_step)
    
    observed_values = data['infected'].values
    if fit_target == 'new_cases':
        simulated_values = np.maximum(0, S[:-1] - S[1:])
    else:
        simulated_values = I[:len(observed_values)]
    
    mse = np.mean((simulated_values - observed_values) ** 2)
    
    return mse


def fit_parameters(data: pd.DataFrame, S0: float, I0: float, R0: float, N: float,
                   time_step: float, fit_target: str, optimizer: str = 'auto') -> Tuple[float, float]:
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
        time_step: Duration (days) represented by each time step
        fit_target: 'active' to match I(t), 'new_cases' to match incidence
    
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
    
    def run_local():
        try:
            result = minimize(
                objective_function,
                x0=[initial_beta, initial_nu],
                args=(data, S0, I0, R0, N, time_step, fit_target),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            if result.success:
                return result
        except Exception as e:
            print(f"Local optimization failed: {e}")
        return None
    
    def run_global():
        print("Running global optimization (differential_evolution)...")
        return differential_evolution(
            objective_function,
            bounds=bounds,
            args=(data, S0, I0, R0, N, time_step, fit_target),
            seed=42,
            maxiter=1000,
            popsize=15
        )
    
    best_result = None
    best_method = None
    
    local_result = None
    if optimizer in ('auto', 'local'):
        local_result = run_local()
        if local_result is not None:
            best_result = local_result
            best_method = 'L-BFGS-B'
    
    need_global = optimizer == 'global'
    if optimizer == 'auto' and best_result is not None:
        near_bound = any(
            np.isclose(best_result.x[i], bounds[i][0], atol=1e-4) or
            np.isclose(best_result.x[i], bounds[i][1], atol=1e-4)
            for i in range(len(bounds))
        )
        if near_bound:
            need_global = True
    
    global_result = None
    if need_global or best_result is None:
        global_result = run_global()
        if best_result is None or global_result.fun < best_result.fun:
            best_result = global_result
            best_method = 'differential_evolution'
    
    if best_result is None:
        raise RuntimeError("Parameter optimization failed")
    
    beta_opt, nu_opt = best_result.x
    beta_interval = beta_opt * time_step
    nu_interval = nu_opt * time_step
    print(
        f"Optimization successful ({best_method}): "
        f"beta = {beta_opt:.6f}/day ({beta_interval:.6f} per {time_step}-day interval), "
        f"nu = {nu_opt:.6f}/day ({nu_interval:.6f} per {time_step}-day interval)"
    )
    return beta_opt, nu_opt


def load_data(csv_path: str, date_column: str = 'date', infected_column: str = 'infected',
              value_window: float = 1.0) -> pd.DataFrame:
    """
    Loads infection data from CSV file.
    
    Expected format:
        date column, infected column
    
    Args:
        csv_path: Path to input CSV file
        date_column: Name of the column containing the timeline values
        infected_column: Name of the column containing infection counts
        value_window: If each row aggregates over `value_window` days, divide by this
    
    Returns:
        DataFrame with 'date' and 'infected' columns
    """
    data = pd.read_csv(csv_path)
    
    # Validate required columns
    missing_columns = [col for col in (date_column, infected_column) if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"CSV must contain the specified columns: missing {', '.join(missing_columns)}"
        )
    
    if value_window <= 0:
        raise ValueError("value_window must be a positive number")
    
    processed = pd.DataFrame({
        'date': pd.to_datetime(data[date_column]),
        'infected': pd.to_numeric(data[infected_column], errors='coerce')
    })
    
    # Remove any rows with missing values and sort chronologically
    processed = processed.dropna(subset=['date', 'infected']).sort_values('date').reset_index(drop=True)
    
    # Convert aggregated window totals to per-day averages if requested
    if value_window != 1.0:
        processed['infected'] = processed['infected'] / value_window
    
    return processed


def export_results(data: pd.DataFrame, S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                   beta: float, nu: float, output_path: str, fit_target: str,
                   time_step: float, simulated_new_cases: np.ndarray = None):
    """
    Exports simulation results to CSV file for Excel plotting.
    
    Columns: date, S, I, R, observed values, beta, nu, optional simulated new cases
    
    Args:
        data: Original data DataFrame with dates
        S, I, R: Simulated arrays from SIR model
        beta, nu: Fitted parameters (per day)
        output_path: Path to output CSV file
        fit_target: Whether observed column represents 'active' or 'new_cases'
        time_step: Duration (days) represented by each time step
        simulated_new_cases: Optional array of simulated new cases
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare export DataFrame
    export_data = pd.DataFrame({
        'date': data['date'].values,
        'time_step_index': np.arange(len(data)),
        'S': S[:len(data)],
        'I': I[:len(data)],
        'R': R[:len(data)],
    })
    
    observed_label = 'observed_new_cases' if fit_target == 'new_cases' else 'observed_I'
    export_data[observed_label] = data['infected'].values
    
    if simulated_new_cases is not None:
        export_data['simulated_new_cases'] = simulated_new_cases[:len(data)]
    
    export_data['beta'] = beta
    export_data['nu'] = nu
    export_data['beta_per_interval'] = beta * time_step
    export_data['nu_per_interval'] = nu * time_step
    
    r0 = beta / nu if nu != 0 else np.nan
    infectious_period_days = 1.0 / nu if nu != 0 else np.nan
    infectious_period_intervals = infectious_period_days / time_step if nu != 0 else np.nan
    
    export_data['R0_per_day'] = r0
    export_data['R0_per_interval'] = export_data['beta_per_interval'] / export_data['nu_per_interval']
    export_data['infectious_period_days'] = infectious_period_days
    export_data['infectious_period_intervals'] = infectious_period_intervals
    
    # Export to CSV
    export_data.to_csv(output_path, index=False)
    print(f"Results exported to: {output_path}")


def plot_results(data: pd.DataFrame, S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                 beta: float, nu: float, save_path: str = None, fit_target: str = 'active',
                 time_step: float = 1.0, simulated_new_cases: np.ndarray = None):
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
    
    if fit_target == 'new_cases' and simulated_new_cases is not None:
        axes[0].plot(dates, simulated_new_cases[:days], 'b-', label='Simulated new cases', linewidth=2)
        axes[0].scatter(dates, data['infected'].values, color='red', marker='o',
                        label='Observed new cases', s=50, alpha=0.7)
        ylabel = f'New cases per {time_step:.0f}-day interval'
        title = f'SIR Model Fit: New Cases (beta = {beta:.4f}, nu = {nu:.4f})'
    else:
        axes[0].plot(dates, I[:days], 'b-', label='Simulated I(t)', linewidth=2)
        axes[0].scatter(dates, data['infected'].values, color='red', marker='o', 
                        label='Observed Infections', s=50, alpha=0.7)
        ylabel = 'Number of Infected'
        title = f'SIR Model Fit: Infections (beta = {beta:.4f}, nu = {nu:.4f})'
    
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
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
    parser.add_argument(
        '--date-column',
        type=str,
        default='date',
        help="Column name to use for the timeline (default: 'date')"
    )
    parser.add_argument(
        '--value-column',
        type=str,
        default='infected',
        help="Column name to use for infection counts (e.g., 'numtotal_last7')"
    )
    parser.add_argument(
        '--value-window',
        type=float,
        default=1.0,
        help='If each row stores a rolling total across this many days, divide by this window (set to 7 for numtotal_last7)'
    )
    parser.add_argument(
        '--time-step',
        type=float,
        default=1.0,
        help='Number of days represented by each row (use 7 for weekly data)'
    )
    parser.add_argument(
        '--fit-target',
        choices=['active', 'new_cases'],
        default='active',
        help="What to fit against: 'active' matches I(t); 'new_cases' matches new infections per time step"
    )
    parser.add_argument(
        '--optimizer',
        choices=['auto', 'local', 'global'],
        default='auto',
        help="Choose optimization strategy: 'auto' runs local and falls back to global if needed"
    )
    
    args = parser.parse_args()
    
    if args.time_step <= 0:
        raise ValueError("--time-step must be positive")
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = load_data(args.data, args.date_column, args.value_column, args.value_window)
    print(f"Loaded {len(data)} data points (time step = {args.time_step} day(s))")
    
    # Set initial conditions
    # I0: first observed infection count (allow fractional values if data is averaged)
    I0 = max(float(data['infected'].iloc[0]), 1e-6)
    # R0: assume no recovered initially
    R0 = 0
    # S0: remaining population
    S0 = float(args.population) - I0 - R0
    
    print(f"\nInitial Conditions:")
    print(f"  Total Population (N): {args.population:,}")
    print(f"  Initial Susceptible (S0): {S0:,.2f}")
    print(f"  Initial Infected (I0): {I0:,.2f}")
    print(f"  Initial Recovered (R0): {R0:,.2f}")
    
    # Fit parameters
    print(f"\nFitting parameters beta and nu...")
    beta, nu = fit_parameters(
        data,
        S0,
        I0,
        R0,
        args.population,
        args.time_step,
        args.fit_target,
        args.optimizer
    )
    beta_interval = beta * args.time_step
    nu_interval = nu * args.time_step
    
    print(f"\nFitted Parameters (per day):")
    print(f"  beta: {beta:.4f} (≈ {beta_interval:.4f} per {args.time_step}-day interval)")
    print(f"  nu:   {nu:.4f} (≈ {nu_interval:.4f} per {args.time_step}-day interval)")
    
    # Run simulation
    print(f"\nRunning SIR simulation...")
    steps = len(data)
    S, I, R = simulate_sir(beta, nu, S0, I0, R0, args.population, steps, args.time_step)
    simulated_new_cases = np.maximum(0, S[:-1] - S[1:])
    
    # Export results
    print(f"\nExporting results...")
    export_results(
        data,
        S,
        I,
        R,
        beta,
        nu,
        args.output,
        args.fit_target,
        args.time_step,
        simulated_new_cases
    )
    
    # Optional plotting
    if args.plot or args.plot_save:
        print(f"\nGenerating plot...")
        plot_results(
            data,
            S,
            I,
            R,
            beta,
            nu,
            args.plot_save,
            args.fit_target,
            args.time_step,
            simulated_new_cases
        )
    
    print(f"\nPipeline completed successfully!")


if __name__ == '__main__':
    main()
