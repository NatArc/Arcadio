import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify

class Areas:
    def __init__(self):
        self.x_sym = symbols('x')
        plt.style.use('dark_background')
    
    def numerical_derivative(self, func_np, x_vals, dx=1e-6):
        deriv = np.zeros_like(x_vals)
        deriv[1:-1] = (func_np(x_vals[2:]) - func_np(x_vals[:-2])) / (2 * dx)
        if len(x_vals) > 1:
            deriv[0] = (func_np(x_vals[1]) - func_np(x_vals[0])) / dx
            deriv[-1] = (func_np(x_vals[-1]) - func_np(x_vals[-2])) / dx
        return deriv
    
    def numerical_integral(self, func_np, x_vals):
        y_vals = func_np(x_vals)
        integral = np.zeros_like(x_vals)
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            integral[i] = integral[i-1] + 0.5 * (y_vals[i] + y_vals[i-1]) * dx
        return integral
    
    def area_between_curves(self, y1, y2, x_vals, a=None, b=None):
        if a is None:
            a = x_vals[0]
        if b is None:
            b = x_vals[-1]
        
        mask = (x_vals >= a) & (x_vals <= b)
        area = np.trapezoid(np.abs(y1[mask] - y2[mask]), x_vals[mask])
        return area
    
    def plot_all_with_areas(self, func_str, x_min=-10, x_max=10, a=None, b=None):
        func_sym = sympify(func_str)
        func_np = lambdify(self.x_sym, func_sym, modules=['numpy'])
        
        x_vals = np.linspace(x_min, x_max, 1000)
        f_x = func_np(x_vals)
        f_prime = self.numerical_derivative(func_np, x_vals)
        f_int = self.numerical_integral(func_np, x_vals)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1: f(x) vs x=0 (area under curve)
        axes[0].plot(x_vals, f_x, 'cyan', linewidth=3, label=f'f(x) = {func_str}')
        if a and b:
            mask = (x_vals >= a) & (x_vals <= b)
            axes[0].fill_between(x_vals[mask], 0, f_x[mask], alpha=0.3, color='cyan', label=f'Area={self.area_between_curves(f_x, np.zeros_like(f_x), x_vals, a, b):.3f}')
        axes[0].axhline(0, color='gray', alpha=0.5)
        axes[0].axvline(0, color='gray', alpha=0.5)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title('f(x) - Area under curve', fontsize=11, color='white')
        
        # 2: f'(x) vs x=0
        axes[1].plot(x_vals, f_prime, 'lime', linewidth=3, label="f'(x)")
        if a and b:
            mask = (x_vals >= a) & (x_vals <= b)
            axes[1].fill_between(x_vals[mask], 0, f_prime[mask], alpha=0.3, color='lime', label=f'Area={self.area_between_curves(f_prime, np.zeros_like(f_prime), x_vals, a, b):.3f}')
        axes[1].axhline(0, color='gray', alpha=0.5)
        axes[1].axvline(0, color='gray', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_title('f\'(x) - Area under derivative', fontsize=11, color='white')
        
        # 3: ∫f(x) vs 0
        axes[2].plot(x_vals, f_int, 'gold', linewidth=3, label='∫f(x)dx')
        axes[2].plot([x_vals[0], x_vals[-1]], [0, 0], 'gray', linewidth=2, label='Antiderivative=0')
        if a and b:
            mask = (x_vals >= a) & (x_vals <= b)
            area_int = self.area_between_curves(f_int, np.zeros_like(f_int), x_vals, a, b)
            axes[2].fill_between(x_vals[mask], 0, f_int[mask], alpha=0.3, color='gold', label=f'Area={area_int:.3f}')
        axes[2].axhline(0, color='gray', alpha=0.5)
        axes[2].axvline(0, color='gray', alpha=0.5)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_title('∫f(x)dx - Area under integral', fontsize=11, color='white')
        
        plt.tight_layout()
        plt.show()
        
        # Summary stats
        if a and b:
            print(f"\nAREAS SUMMARY [{a:.1f}, {b:.1f}]")
            print(f"  f(x) area:     {self.area_between_curves(f_x, np.zeros_like(f_x), x_vals, a, b):.4f}")
            print(f"  f'(x) area:    {self.area_between_curves(f_prime, np.zeros_like(f_prime), x_vals, a, b):.4f}")
            print(f"  ∫f(x) area:    {self.area_between_curves(f_int, np.zeros_like(f_int), x_vals, a, b):.4f}")
