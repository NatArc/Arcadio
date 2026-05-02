from plotter import Plotter
from derivative import Derivative
from integral import Integral
from areas import Areas

def get_x_range():
    while True:
        x_input = input("x-range (default -10 10): ").strip()
        if not x_input: return -10, 10
        try:
            parts = x_input.replace(',', ' ').split()
            if len(parts) == 1: return float(parts[0])-10, float(parts[0])+10
            elif len(parts) == 2: return float(parts[0]), float(parts[1])
        except: print("❌ Examples: '-5 5', '0 10', '2'"); continue

def get_area_range():
    while True:
        a_input = input("Area bounds a b (Enter for full range): ").strip()
        if not a_input: return None, None
        try:
            parts = a_input.replace(',', ' ').split()
            return float(parts[0]), float(parts[1])
        except: print("❌ Examples: '0 2', '-1 3'"); continue

def main():
    print("Calculus-Powered Graph")
    print("1: f(x)")
    print("2: f(x) + f'(x)")
    print("3: f(x) + ∫f(x)")
    print("4: All Graphs + Shaded Areas")
    print("0: Exit")
    
    plotter = Plotter()
    deriv = Derivative()
    integ = Integral()
    areas = Areas()
    
    while True:
        choice = input("\n🎮 Choose (1/2/3/4/0): ").strip()
        
        if choice == '0': 
            print("Bye!")
            break
            
        func = input("📈 f(x): ").strip()
        if not func: continue
        
        x_min, x_max = get_x_range()
        print(f"x = [{x_min:.1f}, {x_max:.1f}]")
        
        if choice == '1':
            plotter.plot_function(func, x_min, x_max)
        elif choice == '2':
            deriv.plot_with_derivative(func, x_min, x_max)
        elif choice == '3':
            integ.plot_with_integral(func, x_min, x_max)
        elif choice == '4':
            a, b = get_area_range()
            areas.plot_all_with_areas(func, x_min, x_max, a, b)
        else:
            print("❌ 1,2,3,4 or 0")

if __name__ == "__main__":
    main()
