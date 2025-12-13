import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq

C_m = 1.0
E_L = -80
g_L = 8
g_Na = 20
g_K = 10
E_Na = 60
E_K = -90
tau_n = 1.0

V_half_m = -20
k_m = 15
V_half_n = -25
k_n = 5

def infV(V,V_half,k):
    return 1 / (1 + np.exp((V_half - V) / k))

def neuron_model(t, state, I_e):
    V, n = state
    dVdt = (-g_K * n * (V - E_K) - g_Na * infV(V,V_half_m,k_m) * (V - E_Na) - g_L * (V - E_L) - I_e) / C_m
    dndt = (infV(V,V_half_n,k_n) - n) / tau_n
    return [dVdt, dndt]

def find_equilibrium_points(I_e):
    V_range = np.linspace(-100, 50, 500) 
    n_nc = infV(V_range, V_half_n, k_n) 
    v_nc_1 = -g_Na * infV(V_range, V_half_m, k_m) * (V_range - E_Na) - g_L * (V_range - E_L) - I_e
    v_nc_2 = g_K * (V_range - E_K)
    v_nc = v_nc_1 / v_nc_2
    v_nc = np.nan_to_num(v_nc, nan=10, posinf=10, neginf=-10)
    
    difference = n_nc - v_nc
    equilibrium_points = []

    for i in range(len(difference) - 1):
        if difference[i] * difference[i + 1] <= 0: 
            def equations(V):
                n_eq = infV(V, V_half_n, k_n)
                v_num = -g_Na * infV(V, V_half_m, k_m) * (V - E_Na) - g_L * (V - E_L) - I_e
                v_eq = v_num / (g_K * (V - E_K))
                return n_eq - v_eq
            
            try:
                V_sol = brentq(equations, V_range[i], V_range[i+1])
                n_sol = infV(V_sol, V_half_n, k_n)

                if (-100 <= V_sol <= 50 and 0 <= n_sol <= 1 and 
                    abs(equations(V_sol)) < 1e-6):
                    
                    equilibrium_points.append((V_sol, n_sol))
                    
            except (ValueError, ZeroDivisionError):
                continue
    
    return equilibrium_points
        
def plot_nullclines(I_e_values):
    V_range = np.linspace(-100, 50, 1000)
    fig, axes = plt.subplots(1, len(I_e_values), figsize=(15, 5))
    if len(I_e_values) == 1:
        axes = [axes]
    
    for idx, I_e in enumerate(I_e_values):
        ax = axes[idx]
        n_nullcline = infV(V_range,V_half_n,k_n)
        V_nullcline_1 = -g_Na * infV(V_range,V_half_m,k_m) * (V_range - E_Na) - g_L * (V_range - E_L) - I_e
        V_nullcline_2 = g_K * (V_range - E_K)
        V_nullcline = V_nullcline_1 / V_nullcline_2
        V_nullcline = np.nan_to_num(V_nullcline, nan=10, posinf=10, neginf=-10)

        ax.plot(V_range, n_nullcline, 'b-', label='n-nullcline (dn/dt=0)', linewidth=2)
        ax.plot(V_range, V_nullcline, 'r-', label='V-nullcline (dV/dt=0)', linewidth=2)
        
       
        equilibria = find_equilibrium_points(I_e)
        for point in equilibria:
            ax.plot(point[0], point[1], 'ko', markersize=8, markeredgewidth=2,markerfacecolor='none', label='Equilibrium' if point == equilibria[0] else "")
        
        ax.set_xlabel('Membrane Potential V (mV)')
        ax.set_ylabel('Gating Variable n')
        ax.set_title(f'I_e = {I_e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlim(-100, 50)
    
    plt.tight_layout()
    plt.show()
    return equilibria
    
def plot_vector_field(I_e):
    V_range = np.linspace(-100, 50, 20)
    n_range = np.linspace(0, 1, 20)  
    V_grid, n_grid = np.meshgrid(V_range, n_range)
    
    dVdt_grid = np.zeros_like(V_grid)
    dndt_grid = np.zeros_like(n_grid)
    
    for i in range(len(V_range)):
        for j in range(len(n_range)):
            derivatives = neuron_model(0, [V_grid[j,i], n_grid[j,i]], I_e)
            dVdt_grid[j,i] = derivatives[0]
            dndt_grid[j,i] = derivatives[1]
    
    plt.figure(figsize=(10, 8))
    plt.streamplot(V_grid, n_grid, dVdt_grid, dndt_grid, color='black', density=1.5)
    
    V_fine = np.linspace(-100, 50, 300)
    n_nullcline = infV(V_fine,V_half_n,k_n)
    V_nullcline_numerator = -g_Na * infV(V_fine,V_half_m,k_m) * (V_fine - E_Na) - g_L * (V_fine - E_L) - I_e
    V_nullcline = V_nullcline_numerator / (g_K * (V_fine - E_K))
    V_nullcline = np.nan_to_num(V_nullcline, nan=10, posinf=10, neginf=-10)
    
    plt.plot(V_fine, n_nullcline, 'b-', linewidth=3, label='n-nullcline')
    plt.plot(V_fine, V_nullcline, 'r-', linewidth=3, label='V-nullcline')
   
  
    equilibria = find_equilibrium_points(I_e)
    for point in equilibria:
        plt.plot(point[0], point[1], 'ko', markersize=8, markeredgewidth=2, 
                markerfacecolor='none')
    
    plt.xlabel('Membrane Potential V (mV)')
    plt.ylabel('Gating Variable n')
    plt.title(f'Phase Plane with Vector Field (I_e = {I_e})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.show()

    
print("\n(a) nulliclines")
I_e_values = [-10, 0, 10, 20]
for I_e in I_e_values:
    equilibria = plot_nullclines([I_e])
    print(f"I_e = {I_e}: number of equlibrium =  {len(equilibria)} ")
    for point in equilibria:
        print(f"  position : V = {point[0]:.2f} mV, n = {point[1]:.3f}")
    
print("\n(b) vector field")
for I_e in I_e_values: 
    plot_vector_field(I_e)
 
print("\n dynamics")
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)

initial_conditions = [(-70, 0.1), (-50, 0.3), (-30, 0.6)]
  
plt.figure(figsize=(12, 8))
for V0, n0 in initial_conditions:
    sol = solve_ivp(neuron_model, t_span, [V0, n0], args=(0,), t_eval=t_eval, method='RK45')
        
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, sol.y[0], label=f'V0={V0}, n0={n0}')
        
    plt.subplot(2, 1, 2)
    plt.plot(sol.t, sol.y[1], label=f'V0={V0}, n0={n0}')

plt.subplot(2, 1, 1)
plt.ylabel('Membrane Potential V (mV)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.xlabel('Time (ms)')
plt.ylabel('Gating Variable n')
plt.legend()
plt.grid(True, alpha=0.3)
  
plt.suptitle('Neuron Dynamics with Different Initial Conditions (I_e = 0)')
plt.show()
    
