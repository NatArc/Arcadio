import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify

class Integral:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('dark_background')
    
    def numerical_integral(self, func_np, x_vals):
        """Compute cumulative integral using trapezoidal rule"""
        y_vals = func_np(x_vals)
        integral = np.zeros_like(x_vals)
        integral[0] = 0
        
        # Trapezoidal rule: cumulative
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            integral[i] = integral[i-1] + 0.5 * (y_vals[i] + y_vals[i-1]) * dx
            
        return integral
    
    def plot_with_integral(self, func_str, x_min=-10, x_max=10):
        """Plot f(x) + ∫f(x)dx side-by-side"""
        func_sym = sympify(func_str)
        func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
        
        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = func_np(x_vals)
        y_integral = self.numerical_integral(func_np, x_vals)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Original f(x)
        ax1.plot(x_vals, y_vals, 'cyan', linewidth=3, label=f'f(x) = {func_str}')
        ax1.axhline(0, color='gray', alpha=0.5)
        ax1.axvline(0, color='gray', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Original Function f(x)', fontsize=11, color='white')
        
        # Integral ∫f(x)dx
        ax2.plot(x_vals, y_integral, 'gold', linewidth=3, label='∫f(x)dx (Numerical)')
        ax2.axhline(0, color='gray', alpha=0.5)
        ax2.axvline(0, color='gray', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('Indefinite Integral ∫f(x)dx', fontsize=11, color='white')
        
        plt.tight_layout()
        plt.show()
        
        # Print definite integral value
        total_area = y_integral[-1]
        print(f"📏 Total integral from {x_min} to {x_max}: {total_area:.4f}")