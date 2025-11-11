# SIR Model Simulation Pipeline

This pipeline reads daily infection data from a CSV file, fits SIR model parameters (β and ν) to the data using optimization, and exports simulation results ready for Excel plotting.

## Features

- **SIR Model Implementation**: Uses Euler's Method with step size h=1 (1 day)
- **Parameter Fitting**: Automatically fits β (infection rate) and ν (recovery rate) to minimize MSE
- **Excel-Ready Export**: CSV output with all simulation data and fitted parameters
- **Optional Visualization**: Matplotlib plotting for quick visual verification

## Installation

Required packages:
```bash
pip install numpy pandas scipy matplotlib
```

## Usage

### Basic Usage
```bash
python sir_pipeline.py --data data/cases.csv --population 1000000
```

### With Plotting
```bash
python sir_pipeline.py --data data/cases.csv --population 1000000 --plot
```

### Save Plot
```bash
python sir_pipeline.py --data data/cases.csv --population 1000000 --plot --plot-save output/sir_plot.png
```

### Custom Output Path
```bash
python sir_pipeline.py --data data/cases.csv --population 1000000 --output output/my_results.csv
```

## Input CSV Format

The input CSV file must have the following format:

```csv
date,infected
2024-01-01,10
2024-01-02,15
2024-01-03,23
...
```

- `date`: Date in any format parseable by pandas (YYYY-MM-DD recommended)
- `infected`: Number of infected individuals on that date

## Output CSV Format

The output CSV (`output/sir_simulation.csv`) contains:

- `date`: Date from input data
- `S`: Simulated susceptible population
- `I`: Simulated infected population
- `R`: Simulated recovered population
- `observed_I`: Original observed infection data
- `beta`: Fitted infection rate parameter
- `nu`: Fitted recovery rate parameter

This format is ready for direct import into Excel for plotting.

## SIR Model Equations

The SIR model uses the following differential equations:

```
dS/dt = -β * S * I / N
dI/dt = β * S * I / N - ν * I
dR/dt = ν * I
```

Where:
- **S**: Susceptible population
- **I**: Infected population
- **R**: Recovered population
- **N**: Total population (S + I + R)
- **β**: Infection rate (transmission rate)
- **ν**: Recovery rate

## Euler's Method

The simulation uses Euler's Method with step size h=1 day:

```
S(t+1) = S(t) + h * dS/dt
I(t+1) = I(t) + h * dI/dt
R(t+1) = R(t) + h * dR/dt
```

## Parameter Fitting

The pipeline uses `scipy.optimize` to find optimal β and ν values that minimize the mean squared error (MSE) between simulated and observed infection data. It first tries local optimization (L-BFGS-B) and falls back to global optimization (differential_evolution) if needed.

## Example

A sample dataset is provided in `data/cases.csv`. Run:

```bash
python sir_pipeline.py --data data/cases.csv --population 1000000 --plot
```

This will:
1. Load the data
2. Fit β and ν parameters
3. Run the simulation
4. Export results to `output/sir_simulation.csv`
5. Display a plot comparing simulated vs observed infections

