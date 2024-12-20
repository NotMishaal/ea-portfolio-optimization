# Financial Portfolio Optimization Using ARIMA Forecasting and Evolutionary Algorithms

This project demonstrates an end-to-end solution for robust portfolio optimization using time-series forecasting (ARIMA) and a Genetic Algorithm (GA). It addresses the complexity of financial portfolio allocation under uncertainty by integrating predictive modeling, covariance-based risk assessment, and evolutionary search.

## Overview

Given a set of stocks and historical price data, the goal is to determine the allocation of capital across these assets to maximize expected returns while controlling for risk.

Approach:

1. Data Preprocessing: Load and clean historical price data, then compute daily returns.
2. Predictive Modeling (ARIMA): Forecast one-step-ahead returns for each asset.
3. Risk Estimation (Covariance): Compute a covariance matrix from historical returns to measure inter-asset risk.
4. Genetic Algorithm (GA):

   Represent candidate solutions as portfolios (weight vectors).
   Use a Sharpe ratio-based fitness function informed by ARIMA forecasts and covariance risk.
   Apply tournament selection, crossover, and mutation to evolve solutions over multiple generations.

5. Validation: Compare the GA-optimized portfolio against an equal-weighted benchmark to demonstrate improved risk-adjusted performance.

## Repository Structure

- `data/`

  Directory for storing input datasets.

- `ml/`

  Directory for storing machine learning related code.

- `ga/`

  Directory for storing genetic algorithm related code.

- `gui.py`

  Script to run GUI application of the portfolio optimizer.

- `main.ipynb`

  Main notebook for running and testing the project.

- `requirements.txt`

  Contains the Python dependencies required to run the project.

## Requirements

Project was ran on Python 3.10.16

Install dependencies using

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare Data

   Download the dataset from [New York Stock Exchange (NYSE) on Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse)

2. Run Portfolio Optimizer

   - Run `python gui.py` to start GUI application
   - The script will automatically:

     - Preprocess the data.
     - Fit ARIMA models to predict next-step returns.
     - Compute the covariance matrix.
     - Run the GA to find an optimal portfolio allocation.
     - Print the best-found solution and compare it against an equal-weighted benchmark.

   - You can manually adjust the following settings and analyze the results:

     - Stocks to include in the portfolio (default: all).
     - Forecasting Method (ARIMA / default)
     - Normalize Returns (default: False)
     - Population Size (default: 50)
     - Generations (default: 20)
     - Tournament Size (default: 3)
     - Crossover Probability (default: 0.7)
     - Mutation Probability (default: 0.2)
     - Risk-Free Rate (default: 0.02)

3. Analyze Results

   - The application will visualize the evolution of best fitness over generations.
   - Key metrics, such as Sharpe Ratio of the GA, and stock weights for the portfolio are displayed.

## Validation Strategy

The project validates the GA’s effectiveness by comparing its optimized portfolio against an equal-weighted benchmark. The consistent outperformance of the GA solution in terms of Sharpe ratio demonstrates the value of combining predictive modeling with evolutionary optimization.
