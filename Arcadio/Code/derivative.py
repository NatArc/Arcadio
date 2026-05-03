import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify

class Derivative:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('dark_background')
    
    def numerical_derivative(self, func_np, x_vals):
        h = x_vals[1] - x_vals[0]
        deriv = np.zeros_like(x_vals)
        deriv[1:-1] = (func_np(x_vals[2:]) - func_np(x_vals[:-2])) / (2 * h)
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]
        return deriv
    
    def plot_with_derivative(self, func_str, x_min=-10, x_max=10):
        try:
            func_sym = sympify(func_str)
            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
            
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = func_np(x_vals)
            y_deriv = self.numerical_derivative(func_np, x_vals)

            y_vals = np.nan_to_num(y_vals, nan=0.0)
            y_deriv = np.nan_to_num(y_deriv, nan=0.0)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1.plot(x_vals, y_vals, 'cyan', linewidth=3, label=f'f(x) = {func_str}')
            ax1.axhline(0, color='gray', alpha=0.5)
            ax1.axvline(0, color='gray', alpha=0.5)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title('Original f(x)', fontsize=11, color='white')
            
            ax2.plot(x_vals, y_deriv, 'lime', linewidth=3, label="f'(x)")
            ax2.axhline(0, color='gray', alpha=0.5)
            ax2.axvline(0, color='gray', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title("Derivative f'(x)", fontsize=11, color='white')
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error: {e}")