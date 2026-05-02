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

        # Clean invalid values
        y_vals = np.array(y_vals, dtype=float)
        y_vals[~np.isfinite(y_vals)] = 0

        integral = np.zeros_like(x_vals)
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            integral[i] = integral[i-1] + 0.5 * (y_vals[i] + y_vals[i-1]) * dx
        return integral
    
    def area_between_curves(self, y1, y2, x_vals, a=None, b=None):
        if a is None: a = x_vals[0]
        if b is None: b = x_vals[-1]
        mask = (x_vals >= a) & (x_vals <= b)
        return np.trapezoid(np.abs(y1[mask] - y2[mask]), x_vals[mask])
    
    def plot_all_with_areas(self, func_str, x_min=-10, x_max=10, a=None, b=None):
        try:
            func_str = func_str.replace('**', '^')  # Fix sympy
            func_sym = sympify(func_str)
            func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
            
            x_vals = np.linspace(x_min, x_max, 500)
            f_x = func_np(x_vals)
            f_prime = self.numerical_derivative(func_np, x_vals)
            f_int = self.numerical_integral(func_np, x_vals)

            # Clean NaN values
            f_x = np.nan_to_num(f_x, nan=0.0)
            f_prime = np.nan_to_num(f_prime, nan=0.0)
            f_int = np.nan_to_num(f_int, nan=0.0)
            
            plt.figure(figsize=(12, 10))
            
            # 1. f(x) - CYAN SHADING (WORKS)
            plt.subplot(3,1,1)
            plt.plot(x_vals, f_x, 'cyan', linewidth=3, label=f'f(x)={func_str}')
            if a is not None and b is not None:
                mask = (x_vals >= a) & (x_vals <= b)
                area_f = np.trapezoid(np.abs(f_x[mask]), x_vals[mask])
                plt.fill_between(x_vals[mask], 0, f_x[mask], alpha=0.5, color='cyan')
                plt.title(f'CYAN f(x) Area[{a:.1f},{b:.1f}]={area_f:.3f}', fontsize=14, color='cyan')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 2. f'(x) - LIME SHADING (NEW!)
            plt.subplot(3,1,2)
            plt.plot(x_vals, f_prime, 'lime', linewidth=3, label="f'(x)")
            if a is not None and b is not None:
                mask = (x_vals >= a) & (x_vals <= b)
                area_p = np.trapezoid(np.abs(f_prime[mask]), x_vals[mask])
                plt.fill_between(x_vals[mask], 0, np.abs(f_prime[mask]), alpha=0.5, color='lime')
                plt.title(f'LIME f\'(x) Area[{a:.1f},{b:.1f}]={area_p:.3f}', fontsize=14, color='lime')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 3. Integral - GOLD SHADING (NEW!)
            plt.subplot(3,1,3)
            plt.plot(x_vals, f_int, 'gold', linewidth=3, label='∫f(x)dx')
            if a is not None and b is not None:
                mask = (x_vals >= a) & (x_vals <= b)
                area_i = np.trapezoid(f_int[mask], x_vals[mask])
                plt.fill_between(x_vals[mask], 0, f_int[mask], alpha=0.5, color='gold')
                plt.title(f'GOLD ∫f(x) Area[{a:.1f},{b:.1f}]={area_i:.3f}', fontsize=14, color='gold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            if a is not None and b is not None:
                print(f"\nALL SHADING:")
                print(f"  f(x):   {area_f:.3f}")
                print(f"  f'(x):  {area_p:.3f}") 
                print(f"  ∫f(x):  {area_i:.3f}")
                
        except Exception as e:
            print(f"❌ Mode 4 Error: {e}")
