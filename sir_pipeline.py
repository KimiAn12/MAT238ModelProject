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


def objective_function(params: np.ndarray, data: pd.DataFrame, S0: int, I0: int, R0: int, N: int, use_cumulative: bool = True) -> float:
    """
    Objective function for parameter fitting.
    Calculates mean squared error between simulated and observed infection data.
    
    Args:
        params: Array [beta, nu] to be optimized
        data: DataFrame with 'date' and 'infected' columns (or 'cumulative_cases')
        S0, I0, R0: Initial conditions
        N: Total population
        use_cumulative: If True, compare (I + R) to cumulative cases.
                       If False, compare I to active cases.
    
    Returns:
        Mean squared error (MSE) between simulated and observed data
    """
    beta, nu = params
    
    # Number of days to simulate (based on data length)
    days = len(data)
    
    # Run simulation
    S, I, R = simulate_sir(beta, nu, S0, I0, R0, N, days)
    
    # Get observed data
    if use_cumulative and 'cumulative_cases' in data.columns:
        # Compare cumulative infections (I + R) to observed cumulative cases
        simulated_cumulative = (I + R)[:len(data)]
        observed = data['cumulative_cases'].values
    else:
        # Compare active infections I to observed active cases
        simulated_cumulative = I[:len(data)]
        observed = data['infected'].values
    
    # Calculate mean squared error
    mse = np.mean((simulated_cumulative - observed) ** 2)
    
    return mse


def fit_parameters(data: pd.DataFrame, S0: int, I0: int, R0: int, N: int, use_cumulative: bool = True) -> Tuple[float, float]:
    """
    Fits SIR model parameters β and ν to minimize MSE between simulated and observed data.
    
    Uses scipy.optimize.minimize with L-BFGS-B method, with fallback to
    differential_evolution for global optimization if needed.
    
    Args:
        data: DataFrame with 'date' and 'infected' columns (or 'cumulative_cases')
        S0: Initial susceptible population
        I0: Initial infected population
        R0: Initial recovered population
        N: Total population
        use_cumulative: If True, fit to cumulative cases (I + R).
                       If False, fit to active cases (I).
    
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
            args=(data, S0, I0, R0, N, use_cumulative),
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
        args=(data, S0, I0, R0, N, use_cumulative),
        seed=42,
        maxiter=1000,
        popsize=15
    )
    
    beta_opt, nu_opt = result.x
    print(f"Optimization successful (differential_evolution): beta = {beta_opt:.6f}, nu = {nu_opt:.6f}")
    return beta_opt, nu_opt


def load_data(csv_path: str, use_cumulative: bool = True) -> pd.DataFrame:
    """
    Loads infection data from CSV file.
    
    Supports two formats:
    1. Simple format: date, infected
    2. Ontario COVID-19 format: date, totalcases, numdeaths, etc.
    
    Args:
        csv_path: Path to input CSV file
        use_cumulative: If True, use cumulative cases for SIR model fitting.
                       If False, calculate active cases from cumulative data.
    
    Returns:
        DataFrame with 'date' and 'infected' columns
        If Ontario format detected, also includes 'cumulative_cases' and 'deaths'
    """
    data = pd.read_csv(csv_path)
    
    # Convert date column to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    else:
        raise ValueError("CSV must contain 'date' column")
    
    # Check if this is the Ontario COVID-19 format
    if 'totalcases' in data.columns:
        print("Detected Ontario COVID-19 format (totalcases column found)")
        
        # Handle missing values (some rows have '-' for recent data)
        data['totalcases'] = pd.to_numeric(data['totalcases'], errors='coerce')
        if 'numdeaths' in data.columns:
            data['numdeaths'] = pd.to_numeric(data['numdeaths'], errors='coerce').fillna(0)
        else:
            data['numdeaths'] = 0
        
        # Remove rows with missing totalcases (incomplete data)
        data = data.dropna(subset=['totalcases'])
        data = data[data['totalcases'] >= 0]  # Remove negative values
        
        # Sort by date to ensure chronological order
        data = data.sort_values('date').reset_index(drop=True)
        
        if use_cumulative:
            # Use cumulative cases as (I + R) in SIR model
            # For fitting, we'll compare I(t) + R(t) to totalcases
            data['infected'] = data['totalcases'].values
            data['cumulative_cases'] = data['totalcases'].values
            print(f"Using cumulative cases (I+R) for SIR model fitting")
        else:
            # Estimate active cases: cumulative cases - recovered - deaths
            # This is an approximation since we don't have exact recovery data
            # Assume recovery time of ~14 days on average
            data['cumulative_cases'] = data['totalcases'].values
            
            # Calculate active cases by assuming cases recover after recovery_period days
            # This is a simplified approach - for more accuracy, use a moving window
            recovery_period = 14  # days
            active_cases = []
            for i in range(len(data)):
                # Active cases ≈ cases from last recovery_period days
                current_date = data.iloc[i]['date']
                cutoff_date = current_date - pd.Timedelta(days=recovery_period)
                mask = (data['date'] > cutoff_date) & (data['date'] <= current_date)
                recent_cases = data[mask]
                if len(recent_cases) > 0:
                    # Estimate active: new cases in recovery period
                    if i >= recovery_period:
                        active = data.iloc[i]['totalcases'] - data.iloc[i - recovery_period]['totalcases']
                    else:
                        active = data.iloc[i]['totalcases']
                else:
                    active = data.iloc[i]['totalcases']
                active_cases.append(max(0, active))
            
            data['infected'] = active_cases
            print(f"Estimated active cases using {recovery_period}-day recovery period")
        
        print(f"Data range: {data['date'].min()} to {data['date'].max()}")
        print(f"Total data points: {len(data)}")
        print(f"Max cumulative cases: {data['cumulative_cases'].max():,.0f}")
        if 'numdeaths' in data.columns:
            print(f"Max deaths: {data['numdeaths'].max():,.0f}")
    
    elif 'infected' in data.columns:
        # Simple format: date, infected
        print("Detected simple format (infected column found)")
        data['infected'] = pd.to_numeric(data['infected'], errors='coerce')
        data = data.dropna(subset=['infected'])
        data = data[data['infected'] >= 0]
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Create cumulative_cases column for consistency
        data['cumulative_cases'] = data['infected'].cumsum()
    else:
        raise ValueError("CSV must contain either 'infected' or 'totalcases' column")
    
    # Ensure we have valid data
    if len(data) == 0:
        raise ValueError("No valid data found after processing")
    
    return data


def export_results(data: pd.DataFrame, S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                   beta: float, nu: float, output_path: str, use_cumulative: bool = True):
    """
    Exports simulation results to CSV file for Excel plotting.
    
    Columns: date, S, I, R, observed_I, beta, nu
    
    Args:
        data: Original data DataFrame with dates
        S, I, R: Simulated arrays from SIR model
        beta, nu: Fitted parameters
        output_path: Path to output CSV file
        use_cumulative: If True, use cumulative cases as observed_I
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine observed_I based on data type
    # For cumulative cases format, use cumulative cases as observed_I
    # For active cases format, use infected as observed_I
    if use_cumulative and 'cumulative_cases' in data.columns:
        observed_I = data['cumulative_cases'].values
    else:
        observed_I = data['infected'].values
    
    # Prepare export DataFrame in simple format matching ex_output.csv
    export_data = pd.DataFrame({
        'date': data['date'].values,
        'S': S[:len(data)],
        'I': I[:len(data)],
        'R': R[:len(data)],
        'observed_I': observed_I,
        'beta': beta,
        'nu': nu
    })
    
    # Export to CSV
    export_data.to_csv(output_path, index=False)
    print(f"Results exported to: {output_path}")


def plot_results(data: pd.DataFrame, S: np.ndarray, I: np.ndarray, R: np.ndarray, 
                 beta: float, nu: float, save_path: str = None, use_cumulative: bool = True):
    """
    Plots simulated vs observed infection curves.
    
    Args:
        data: Original data DataFrame
        S, I, R: Simulated arrays
        beta, nu: Fitted parameters
        save_path: Optional path to save the plot
        use_cumulative: If True, plot cumulative cases comparison
    """
    days = len(data)
    dates = data['date'].values
    
    # Determine number of subplots based on data availability
    if use_cumulative and 'cumulative_cases' in data.columns:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative Infections (I + R) vs Observed Cumulative Cases
    if use_cumulative and 'cumulative_cases' in data.columns:
        axes[0].plot(dates, (I + R)[:days], 'b-', label='Simulated Cumulative (I+R)', linewidth=2)
        axes[0].scatter(dates, data['cumulative_cases'].values, color='red', marker='o', 
                        label='Observed Cumulative Cases', s=50, alpha=0.7)
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Cumulative Infections')
        axes[0].set_title(f'SIR Model Fit: Cumulative Infections (beta = {beta:.4f}, nu = {nu:.4f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        plot_idx = 1
    else:
        plot_idx = 0
    
    # Plot 2: Active Infections (I) vs Observed
    axes[plot_idx].plot(dates, I[:days], 'b-', label='Simulated I(t)', linewidth=2)
    if not use_cumulative or 'cumulative_cases' not in data.columns:
        axes[plot_idx].scatter(dates, data['infected'].values, color='red', marker='o', 
                               label='Observed Infections', s=50, alpha=0.7)
    axes[plot_idx].set_xlabel('Date')
    axes[plot_idx].set_ylabel('Number of Infected')
    axes[plot_idx].set_title(f'SIR Model: Active Infections (beta = {beta:.4f}, nu = {nu:.4f})')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].tick_params(axis='x', rotation=45)
    
    # Plot 3: All compartments (S, I, R)
    axes[plot_idx + 1].plot(dates, S[:days], 'g-', label='Susceptible (S)', linewidth=2)
    axes[plot_idx + 1].plot(dates, I[:days], 'b-', label='Infected (I)', linewidth=2)
    axes[plot_idx + 1].plot(dates, R[:days], 'r-', label='Recovered (R)', linewidth=2)
    axes[plot_idx + 1].set_xlabel('Date')
    axes[plot_idx + 1].set_ylabel('Population')
    axes[plot_idx + 1].set_title('SIR Model: All Compartments')
    axes[plot_idx + 1].legend()
    axes[plot_idx + 1].grid(True, alpha=0.3)
    axes[plot_idx + 1].tick_params(axis='x', rotation=45)
    
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
        default=None,
        help='Total population N (default: auto-detect for Ontario, or 1000000)'
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
        '--use-cumulative',
        action='store_true',
        help='Use cumulative cases for fitting (I+R) - auto-enabled for Ontario data'
    )
    parser.add_argument(
        '--use-active',
        action='store_true',
        help='Use active cases for fitting instead of cumulative (overrides --use-cumulative)'
    )
    
    args = parser.parse_args()
    
    # Load data (temporarily with default use_cumulative to check format)
    print(f"Loading data from: {args.data}")
    data_temp = pd.read_csv(args.data)
    
    # Determine if we should use cumulative cases
    # Auto-detect: if Ontario format (has totalcases), use cumulative by default
    is_ontario_format = 'totalcases' in data_temp.columns
    if args.use_active:
        use_cumulative = False
    elif args.use_cumulative:
        use_cumulative = True
    else:
        # Auto-detect: use cumulative if Ontario format
        use_cumulative = is_ontario_format
    
    # Now load with correct setting
    data = load_data(args.data, use_cumulative=use_cumulative)
    print(f"Loaded {len(data)} data points")
    
    # Auto-detect population if not specified
    if args.population is None:
        # Check if this is Ontario data
        if 'prname' in data.columns:
            ontario_rows = data['prname'].str.contains('Ontario', case=False, na=False)
            if ontario_rows.any():
                # Ontario population (approximately 14.5-15 million during 2020-2024)
                args.population = 15000000
                print(f"Auto-detected Ontario data, using population: {args.population:,}")
            else:
                args.population = 1000000
                print(f"Using default population: {args.population:,}")
        elif is_ontario_format:
            # Likely Ontario data based on format
            args.population = 15000000
            print(f"Detected Ontario COVID-19 format, using population: {args.population:,}")
        else:
            args.population = 1000000
            print(f"Using default population: {args.population:,}")
    
    # Set initial conditions
    if use_cumulative and 'cumulative_cases' in data.columns:
        # For cumulative cases: I0 + R0 = first cumulative case
        first_cumulative = int(data['cumulative_cases'].iloc[0])
        
        # For early epidemic: assume most cases are still active
        # Use deaths for R0 if available, otherwise assume minimal recovery
        if 'numdeaths' in data.columns:
            initial_deaths = int(data['numdeaths'].iloc[0])
        else:
            initial_deaths = 0
        
        # At the start of an epidemic, most cases are active
        # For SIR model: I0 + R0 = first_cumulative
        # R0 includes deaths (and recoveries, but at start mostly deaths)
        if first_cumulative > 0:
            # Use deaths as initial R0 (recovered/removed)
            R0 = min(initial_deaths, first_cumulative - 1)  # Ensure at least 1 infected
            I0 = first_cumulative - R0
            # Ensure I0 is at least 1 to allow the epidemic to progress
            if I0 < 1:
                I0 = 1
                R0 = first_cumulative - I0
        else:
            # No cases yet - start with 1 infected to begin simulation
            I0 = 1
            R0 = 0
            first_cumulative = 1  # Adjust for consistency
    else:
        # For active cases: use first observed infection count
        I0 = max(1, int(data['infected'].iloc[0]))
        R0 = 0
    
    # S0: remaining population
    S0 = args.population - I0 - R0
    
    # Ensure S0 is non-negative
    if S0 < 0:
        print(f"Warning: S0 is negative ({S0}). Adjusting population or initial conditions.")
        # Adjust I0 to fit within population
        I0 = min(I0, args.population - R0)
        S0 = args.population - I0 - R0
    
    print(f"\nInitial Conditions:")
    print(f"  Total Population (N): {args.population:,}")
    print(f"  Initial Susceptible (S0): {S0:,}")
    print(f"  Initial Infected (I0): {I0:,}")
    print(f"  Initial Recovered (R0): {R0:,}")
    if use_cumulative and 'cumulative_cases' in data.columns:
        print(f"  First Observed Cumulative Cases: {data['cumulative_cases'].iloc[0]:,.0f}")
    
    # Fit parameters
    print(f"\nFitting parameters beta and nu...")
    print(f"  Fitting method: {'Cumulative cases (I+R)' if use_cumulative else 'Active cases (I)'}")
    beta, nu = fit_parameters(data, S0, I0, R0, args.population, use_cumulative=use_cumulative)
    
    # Run simulation
    print(f"\nRunning SIR simulation...")
    days = len(data)
    S, I, R = simulate_sir(beta, nu, S0, I0, R0, args.population, days)
    
    # Export results
    print(f"\nExporting results...")
    export_results(data, S, I, R, beta, nu, args.output, use_cumulative=use_cumulative)
    
    # Optional plotting
    if args.plot or args.plot_save:
        print(f"\nGenerating plot...")
        plot_results(data, S, I, R, beta, nu, args.plot_save, use_cumulative=use_cumulative)
    
    print(f"\nPipeline completed successfully!")
    print(f"\nFitted Parameters:")
    print(f"  Beta (infection rate): {beta:.6f}")
    print(f"  Nu (recovery rate): {nu:.6f}")
    print(f"  R0 (reproduction number): {beta/nu:.4f}")


if __name__ == '__main__':
    main()

