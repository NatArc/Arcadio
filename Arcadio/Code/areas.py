import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify

class Areas:
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
    
    def numerical_integral(self, func_np, x_vals):
        y_vals = func_np(x_vals)
        y_vals = np.nan_to_num(y_vals, nan=0.0)  

        integral = np.zeros_like(x_vals)
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            integral[i] = integral[i-1] + 0.5 * (y_vals[i] + y_vals[i-1]) * dx
        
        integral = integral - integral[0]
        return integral
    
    def definite_integral_value(self, f_int, a, b, x_vals):
        """F(b) - F(a) for definite integral"""
        idx_a = np.argmax(x_vals >= a)
        idx_b = np.argmax(x_vals >= b)
        return f_int[idx_b] - f_int[idx_a]
    
    def plot_all_with_areas(self, func_str, x_min=-10, x_max=10, a=None, b=None):
        try:
            func_sym = sympify(func_str)
            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
            
            x_vals = np.linspace(x_min, x_max, 1000)
            f_x = func_np(x_vals)
            f_prime = self.numerical_derivative(func_np, x_vals)
            f_int = self.numerical_integral(func_np, x_vals)

            f_x = np.nan_to_num(f_x, nan=0.0)
            f_prime = np.nan_to_num(f_prime, nan=0.0)
            f_int = np.nan_to_num(f_int, nan=0.0)
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            ax1 = axes[0]
            ax1.plot(x_vals, f_x, 'cyan', linewidth=3, label=f'f(x)={func_str}')
            if a is not None and b is not None:
                mask = (x_vals >= a) & (x_vals <= b)
                area_f = np.trapezoid(f_x[mask], x_vals[mask])
                ax1.fill_between(x_vals[mask], 0, f_x[mask], alpha=0.4, color='cyan')
                ax1.set_title(f'Area Under Curve: f(x) Area[{a:.1f},{b:.1f}] = {area_f:.4f}', 
                            fontsize=14, color='cyan', pad=20)
            ax1.axhline(0, color='gray', alpha=0.3)
            ax1.axvline(0, color='gray', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2 = axes[1]
            ax2.plot(x_vals, f_prime, 'lime', linewidth=3, label="f'(x)")
            if a is not None and b is not None:
                mask = (x_vals >= a) & (x_vals <= b)
                area_p = np.trapezoid(np.abs(f_prime[mask]), x_vals[mask])
                ax2.fill_between(x_vals[mask], 0, np.abs(f_prime[mask]), alpha=0.4, color='lime')
                ax2.set_title(f'Area Under Derivative: f\'(x) Area[{a:.1f},{b:.1f}] = {area_p:.4f}', 
                            fontsize=14, color='lime', pad=20)
            ax2.axhline(0, color='gray', alpha=0.3)
            ax2.axvline(0, color='gray', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            ax3 = axes[2]
            ax3.plot(x_vals, f_int, 'gold', linewidth=3, label='F(x) = ∫f(x)dx')
            if a is not None and b is not None:
                mask = (x_vals >= a) & (x_vals <= b)
                area_i = self.definite_integral_value(f_int, a, b, x_vals)
                
                ax3.fill_between(x_vals[mask], 0, f_int[mask], alpha=0.4, color='gold')
                ax3.axvline(a, color='white', linestyle='--', alpha=0.7)
                ax3.axvline(b, color='white', linestyle='--', alpha=0.7)
                ax3.set_title(f'Definite Integral: ∫f(x)dx [{a:.1f},{b:.1f}] = F({b:.1f})-F({a:.1f}) = {area_i:.4f}', 
                            fontsize=14, color='gold', pad=20)
            ax3.axhline(0, color='gray', alpha=0.3)
            ax3.axvline(0, color='gray', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88, hspace=0.4)
            plt.show()
            
            print(f"\nAreas Summary:")
            if a is not None and b is not None:
                print(f" Bounds [{a:.1f}, {b:.1f}]:")
                mask = (x_vals >= a) & (x_vals <= b)
                area_f = np.trapezoid(f_x[mask], x_vals[mask])
                area_p = np.trapezoid(np.abs(f_prime[mask]), x_vals[mask])
                area_i = self.definite_integral_value(f_int, a, b, x_vals)
                print(f"    f(x):      {area_f:.4f}")
                print(f"    f'(x):   {area_p:.4f}")
                print(f"    ∫[a,b]f:   {area_i:.4f}")
            else:
                print(f"  Full range [{x_min:.1f}, {x_max:.1f}]:")
                print(f"    f(x):      {np.trapezoid(f_x, x_vals):.4f}")
                print(f"    f'(x):   {np.trapezoid(np.abs(f_prime), x_vals):.4f}")
                            
        except Exception as e:
            print(f"❌ Mode 4 Error: {e}")
