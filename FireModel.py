import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def simulater(I_e, tau=10, tau_a=200, J_a=0.1, T_total=1000, dt=0.1):
    
    n_steps = int(T_total / dt)
    time = np.arange(0, T_total, dt)
    
    V = np.zeros(n_steps)
    I_adapt = np.zeros(n_steps)
    spike_times = []
    
    V[0] = 0
    I_adapt[0] = 0
    
    for t in range(1, n_steps):
        dVdt = (-V[t-1] + I_e + I_adapt[t-1]) / tau
        V[t] = V[t-1] + dVdt * dt
   
        dI_adt = -I_adapt[t-1] / tau_a
        I_adapt[t] = I_adapt[t-1] + dI_adt * dt
  
        if V[t] >= 1: 
            spike_times.append(time[t])
            V[t] = 0  
            I_adapt[t] -= J_a  
    
    return time, V, I_adapt, spike_times

def no_adaptation(I_range, tau=10):
  
    firing_rates = []

    for I_e in I_range:
        if I_e <= 1:  
            firing_rates.append(0)
        else:
            T = tau * np.log(I_e / (I_e - 1))
            firing_rates.append(1000 / T) 
    
    return firing_rates

def with_adaptation(I_range, tau=10, tau_a=200, J_a=0.1):
    
    firing_rates = []
    first_ISI = []
    
    for I_e in I_range:
        time, V, I_adapt, spikes = simulater(I_e, tau=tau, tau_a=tau_a, J_a=J_a, T_total=2000)
     
        if len(spikes) > 10:
            steady_spikes = spikes[5:]
            if len(steady_spikes) > 1:
                avg_isi = np.mean(np.diff(steady_spikes))
                firing_rate = 1000 / avg_isi
            else:
                firing_rate = 0
        else:
            firing_rate = 0
  
        if len(spikes) >= 2:
            first_isi = spikes[1] - spikes[0]
        else:
            first_isi = np.nan
            
        firing_rates.append(firing_rate)
        first_ISI.append(first_isi)
    
    return firing_rates, first_ISI

 
    

print("=" * 50)
print("Integrate and Fire Model with Adaptation")



print("f-I")
I_range = np.linspace(0.5, 3, 20)
    
plt.figure(figsize=(15, 10))
    
f_rates_no_adapt = no_adaptation(I_range)
f_rates_adapt1, first_isi1 = with_adaptation(I_range, J_a=0.1)
f_rates_adapt2, first_isi2 = with_adaptation(I_range, J_a=1.0)
plt.subplot(2, 2, 1)
plt.plot(I_range, f_rates_no_adapt, 'bo-', label='no adaptation', markersize=4)
plt.plot(I_range, f_rates_adapt1, 'ro-', label='Ja=0.1', markersize=4)
plt.plot(I_range, f_rates_adapt2, 'go-', label='Ja=1.0', markersize=4)
plt.xlabel('I_e')
plt.ylabel('Hz')
plt.title('f-I')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
first_isi_rates1 = [1000/isi if not np.isnan(isi) and isi > 0 else 0 for isi in first_isi1]
first_isi_rates2 = [1000/isi if not np.isnan(isi) and isi > 0 else 0 for isi in first_isi2]
plt.plot(I_range, f_rates_adapt1, 'ro-', label='Ja=0.1 stable', markersize=4)
plt.plot(I_range, first_isi_rates1, 'r--', label='Ja=0.1 ISI', markersize=4)
plt.plot(I_range, f_rates_adapt2, 'go-', label='Ja=1.0 sta ble', markersize=4)
plt.plot(I_range, first_isi_rates2, 'g--', label='Ja=1.0 ISI', markersize=4)
plt.xlabel('I_e')
plt.ylabel('Hz')
plt.title('ISI vs f')
plt.legend()
plt.grid(True, alpha=0.3)
    
plt.subplot(2, 2, 3)
time, V, I_adapt, spikes = simulater(I_e=1.2, J_a=0.1, T_total=500)
plt.plot(time, V, 'b-', label='I_e=1.2', linewidth=1)
time, V, I_adapt, spikes = simulater(I_e=2.5, J_a=0.1, T_total=500)
plt.plot(time, V, 'r-', label='I_e=2.5', linewidth=1)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='threshold')
plt.xlabel('t (ms)')
plt.ylabel('V (V)')
plt.title('time order (Ja=0.1)')
plt.legend()
plt.grid(True, alpha=0.3)
    
plt.subplot(2, 2, 4)  
time, V, I_adapt, spikes = simulater(I_e=2.0, J_a=0.1, T_total=500)  
plt.plot(time, V, 'b-', label='V', linewidth=1)
plt.plot(time, I_adapt, 'r-', label='Ia', linewidth=1)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='threshold')
plt.xlabel('t (ms)')
plt.ylabel('A')
plt.title('V vs Ia')
plt.legend()
plt.grid(True, alpha=0.3)
    
plt.tight_layout()
plt.show()

print("membrane")

currents = [1.2, 1.8, 2.5]
Ja_values = [0, 0.1, 1.0]
    
plt.figure(figsize=(15, 10))
    
for i, I_e in enumerate(currents):
    for j, J_a in enumerate(Ja_values):
        plt.subplot(3, 3, i*3 + j + 1)
            
        time, V, I_adapt, spikes = simulater(I_e=I_e, J_a=J_a, T_total=500)
        plt.plot(time, V, linewidth=1)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            
        plt.title(f'I_e={I_e}, Ja={J_a}, spikes={len(spikes)}')
        plt.xlabel('t (ms)')
        plt.ylabel('V (V)')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.5, 1.5)
    
plt.tight_layout()
plt.show()

print("\n test:")
I_test = 2.0
time, V, I_adapt, spikes = simulater(I_test, 0.1, 1000)
print(f"I_e = {I_test}: spikes = {len(spikes)}")
if len(spikes) > 1:
    isi = np.diff(spikes)
    print(f"ISI_list: {isi[:5]}...")  
    print(f"mean_ISI: {np.mean(isi):.2f} ms")
    print(f"f: {1000/np.mean(isi[5:] if len(isi)>5 else isi):.2f} Hz")

print("\n completed")