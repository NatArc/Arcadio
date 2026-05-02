import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify

class Plotter:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = 'black'
    
    def plot_function(self, func_str, x_min=-10, x_max=10):
        try:
            func_sym = sympify(func_str)
            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
        
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = func_np(x_vals)

            # Fix invalid values
            y_vals = np.array(y_vals, dtype=float)
            y_vals[~np.isfinite(y_vals)] = np.nan

            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, y_vals, 'cyan', linewidth=3, label=f'f(x) = {func_str}')
            plt.axhline(0, color='gray', alpha=0.5)
            plt.axvline(0, color='gray', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.title(f'f(x) = {func_str}', fontsize=16, pad=20, color='white')
            plt.xlabel('x', fontsize=14, color='white')
            plt.ylabel('f(x)', fontsize=14, color='white')
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error plotting function: {e}")