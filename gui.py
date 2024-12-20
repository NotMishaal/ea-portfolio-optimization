import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from ga.cov_ga import GeneticOptimizer
from ml.forecasting import calculate_forecast, compute_covariance_matrix, forecast_returns_arima
from ml.data_preprocessing import load_data, compute_returns, normalize_returns

class PortfolioOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data = None
        self.returns = None
        self.selected_tickers = []
        
        # Create main containers
        self.create_input_frame()
        self.create_parameters_frame()
        self.create_visualization_frame()
        self.create_results_frame()
        
    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Data Input", padding="5")
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Button(input_frame, text="Load CSV Data", command=self.load_data).grid(row=0, column=0, padx=5, pady=5)
        self.ticker_entry = ttk.Entry(input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(input_frame, text="Enter tickers (comma-separated)").grid(row=0, column=2, padx=5, pady=5)
        
    def create_parameters_frame(self):
        param_frame = ttk.LabelFrame(self.root, text="Algorithm Parameters", padding="5")
        param_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Population size
        ttk.Label(param_frame, text="Population Size:").grid(row=0, column=0, padx=5, pady=5)
        self.pop_size_var = tk.StringVar(value="50")
        ttk.Entry(param_frame, textvariable=self.pop_size_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Number of generations
        ttk.Label(param_frame, text="Generations:").grid(row=0, column=2, padx=5, pady=5)
        self.ngen_var = tk.StringVar(value="20")
        ttk.Entry(param_frame, textvariable=self.ngen_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Tournament size
        ttk.Label(param_frame, text="Tournament Size:").grid(row=1, column=0, padx=5, pady=5)
        self.tourn_size_var = tk.StringVar(value="3")
        ttk.Entry(param_frame, textvariable=self.tourn_size_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Crossover probability
        ttk.Label(param_frame, text="Crossover Prob:").grid(row=1, column=2, padx=5, pady=5)
        self.cxpb_var = tk.StringVar(value="0.7")
        ttk.Entry(param_frame, textvariable=self.cxpb_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Mutation probability
        ttk.Label(param_frame, text="Mutation Prob:").grid(row=2, column=0, padx=5, pady=5)
        self.mutpb_var = tk.StringVar(value="0.2")
        ttk.Entry(param_frame, textvariable=self.mutpb_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # Risk-free rate
        ttk.Label(param_frame, text="Risk-free Rate:").grid(row=2, column=2, padx=5, pady=5)
        self.rf_rate_var = tk.StringVar(value="0.0")
        ttk.Entry(param_frame, textvariable=self.rf_rate_var, width=10).grid(row=2, column=3, padx=5, pady=5)
        
        # Run button
        ttk.Button(param_frame, text="Optimize Portfolio", command=self.run_optimization).grid(row=3, column=0, columnspan=4, pady=10)
        
    def create_visualization_frame(self):
        self.viz_frame = ttk.LabelFrame(self.root, text="Optimization Progress", padding="5")
        self.viz_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")
        
        # Create matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_results_frame(self):
        self.results_frame = ttk.LabelFrame(self.root, text="Optimization Results", padding="5")
        self.results_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.results_text = tk.Text(self.results_frame, height=10, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_data(self):
        pass
    
    def run_optimization(self):
        pass

def main():
    root = tk.Tk()
    app = PortfolioOptimizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
