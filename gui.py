import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from ga.cov_ga import GeneticOptimizer
from ml.forecasting import calculate_forecast, compute_covariance_matrix, forecast_returns_arima
from ml.data_preprocessing import load_data, compute_returns, normalize_returns

class DarkTheme:
    BG_COLOR = "#2b2b2b"
    FG_COLOR = "#ffffff"
    FRAME_BG = "#363636"
    ENTRY_BG = "#404040"
    BUTTON_BG = "#4a4a4a"
    BUTTON_FG = "#ffffff"
    
    @classmethod
    def apply_theme(cls):
        style = ttk.Style()
        style.theme_use('alt')
        
        style.configure("TLabel", foreground=cls.FG_COLOR, background=cls.FRAME_BG)
        style.configure("TButton", foreground=cls.BUTTON_FG, background=cls.BUTTON_BG)
        style.configure("TEntry", fieldbackground=cls.ENTRY_BG, foreground=cls.FG_COLOR)
        style.configure("TLabelframe", background=cls.FRAME_BG, foreground=cls.FG_COLOR)
        style.configure("TLabelframe.Label", background=cls.FRAME_BG, foreground=cls.FG_COLOR)
        style.configure("TCheckbutton", foreground=cls.FG_COLOR, background=cls.FRAME_BG)
        style.configure("TRadiobutton", foreground=cls.FG_COLOR, background=cls.FRAME_BG)
        
class PortfolioOptimizerGUI:
    def __init__(self, root, data_file="data/prices.csv"):
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.root.geometry("1200x800")
        self.root.configure(bg=DarkTheme.BG_COLOR)
        
        DarkTheme.apply_theme()
        
        # Initialize data
        self.data_file = data_file
        self.data = None
        self.returns = None
        self.selected_tickers = []
        self.all_tickers = []
        
        # Load initial data
        self.initial_data_load()
        
        self.create_input_frame()
        self.create_options_frame()
        self.create_parameters_frame()
        self.create_visualization_frame()
        self.create_results_frame()

    def initial_data_load(self):
        """Load the initial dataset and get all available tickers."""
        try:
            initial_data = load_data(self.data_file)
            self.all_tickers = list(initial_data.columns)
            self.data = initial_data
            self.returns = compute_returns(self.data)
            self.selected_tickers = self.all_tickers
        except Exception as e:
            messagebox.showerror("Error", f"Error loading initial data: {str(e)}")
            self.root.destroy()  # Exit if we can't load the data
        
    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Data Selection", padding="5")
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Ticker entry
        ttk.Label(input_frame, text="Filter Tickers:").grid(row=0, column=0, padx=5, pady=5)
        self.ticker_entry = ttk.Entry(input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Apply filter button
        ttk.Button(input_frame, text="Apply Filter", command=self.filter_tickers).grid(row=0, column=2, padx=5, pady=5)
        
        # Show all button
        ttk.Button(input_frame, text="Show All Tickers", command=self.show_all_tickers).grid(row=0, column=3, padx=5, pady=5)
        
        # Display current selection
        self.ticker_count_label = ttk.Label(input_frame, text=f"Selected: {len(self.selected_tickers)} tickers")
        self.ticker_count_label.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

    def filter_tickers(self):
        """Filter the dataset based on user-entered tickers."""
        tickers = [t.strip().upper() for t in self.ticker_entry.get().split(",") if t.strip()]
        
        if not tickers:
            messagebox.showwarning("Warning", "Please enter at least one ticker")
            return
            
        # Validate tickers
        valid_tickers = [t for t in tickers if t in self.all_tickers]
        invalid_tickers = set(tickers) - set(valid_tickers)
        
        if invalid_tickers:
            messagebox.showwarning("Warning", f"Invalid tickers: {', '.join(invalid_tickers)}")
        
        if valid_tickers:
            try:
                self.data = load_data(self.data_file, valid_tickers)
                self.returns = compute_returns(self.data)
                self.selected_tickers = valid_tickers
                self.ticker_count_label.config(text=f"Selected: {len(self.selected_tickers)} tickers")
                messagebox.showinfo("Success", f"Filtered to {len(valid_tickers)} tickers")
            except Exception as e:
                messagebox.showerror("Error", f"Error filtering data: {str(e)}")
                
    def show_all_tickers(self):
        """Reset to show all available tickers."""
        try:
            self.data = load_data(self.data_file)
            self.returns = compute_returns(self.data)
            self.selected_tickers = self.all_tickers
            self.ticker_entry.delete(0, tk.END)
            self.ticker_count_label.config(text=f"Selected: {len(self.selected_tickers)} tickers")
            messagebox.showinfo("Success", "Showing all available tickers")
        except Exception as e:
            messagebox.showerror("Error", f"Error resetting data: {str(e)}")

    def create_options_frame(self):
        options_frame = ttk.LabelFrame(self.root, text="Analysis Options", padding="5")
        options_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Forecast method selection
        ttk.Label(options_frame, text="Forecast Method:").grid(row=0, column=0, padx=5, pady=5)
        self.forecast_method = tk.StringVar(value="default")
        ttk.Radiobutton(options_frame, text="Default", variable=self.forecast_method, 
                       value="default").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(options_frame, text="ARIMA", variable=self.forecast_method, 
                       value="arima").grid(row=0, column=2, padx=5, pady=5)
        
        # Returns normalization toggle
        ttk.Label(options_frame, text="Returns Processing:").grid(row=1, column=0, padx=5, pady=5)
        self.normalize_returns_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use Normalized Returns", 
                       variable=self.normalize_returns_var).grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        
    def create_parameters_frame(self):
        param_frame = ttk.LabelFrame(self.root, text="Algorithm Parameters", padding="5")
        param_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
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
        self.viz_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nsew")
        
        plt.style.use('dark_background')
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.figure.patch.set_facecolor(DarkTheme.BG_COLOR)
        self.ax.set_facecolor(DarkTheme.FRAME_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_results_frame(self):
        self.results_frame = ttk.LabelFrame(self.root, text="Optimization Results", padding="5")
        self.results_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.results_text = tk.Text(self.results_frame, height=10, width=80, 
                                  bg=DarkTheme.ENTRY_BG, fg=DarkTheme.FG_COLOR,
                                  insertbackground=DarkTheme.FG_COLOR)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                tickers = [t.strip() for t in self.ticker_entry.get().split(",") if t.strip()]
                self.data = load_data(file_path, tickers if tickers else None)
                self.returns = compute_returns(self.data)
                self.selected_tickers = list(self.data.columns)
                messagebox.showinfo("Success", f"Loaded data for {len(self.selected_tickers)} stocks")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def run_optimization(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
            
        try:
            # Process returns based on normalization option
            processed_returns = normalize_returns(self.returns) if self.normalize_returns_var.get() else self.returns
            
            # Calculate expected returns based on selected method
            if self.forecast_method.get() == "arima":
                expected_returns, _ = calculate_forecast(processed_returns)
                expected_returns = expected_returns.values
            else:
                expected_returns = forecast_returns_arima(processed_returns)
            
            # Get parameters from GUI
            optimizer = GeneticOptimizer(
                pop_size=int(self.pop_size_var.get()),
                ngen=int(self.ngen_var.get()),
                tourn_size=int(self.tourn_size_var.get()),
                cxpb=float(self.cxpb_var.get()),
                mutpb=float(self.mutpb_var.get()),
                risk_free_rate=float(self.rf_rate_var.get())
            )
            
            cov_matrix = compute_covariance_matrix(processed_returns)
            
            # Run optimization
            best_portfolio, best_fitness, fitness_history = optimizer.run(
                expected_returns,
                cov_matrix.values,
                self.selected_tickers
            )
            
            # Update visualization
            self.ax.clear()
            self.ax.plot(fitness_history, color='#00ff00')
            self.ax.set_xlabel("Generation")
            self.ax.set_ylabel("Best Sharpe Ratio")
            self.ax.set_title("Optimization Progress")
            self.ax.grid(True, linestyle='--', alpha=0.3)
            self.canvas.draw()
            
            # Update results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Best Sharpe Ratio: {best_fitness:.4f}\n")
            self.results_text.insert(tk.END, f"Forecast Method: {self.forecast_method.get().upper()}\n")
            self.results_text.insert(tk.END, f"Using Normalized Returns: {self.normalize_returns_var.get()}\n\n")
            self.results_text.insert(tk.END, "Optimal Portfolio Weights:\n")
            for ticker, weight in zip(self.selected_tickers, best_portfolio):
                self.results_text.insert(tk.END, f"{ticker}: {weight:.4f}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Error during optimization: {str(e)}")

def main():
    root = tk.Tk()
    app = PortfolioOptimizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()