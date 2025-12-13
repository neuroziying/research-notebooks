import numpy as np
kB = 1.380649e-23
e = 1.60217662e-19
T = 310.15

def GHK(PM,PA,M_in,M_out,A_in,A_out):
    G1 = 0
    G2 = 0
    G = 0
    for i in range(len(M_out)):
        G1 += PM[i]*M_out[i]
        G2 += PM[i]*M_in[i]
    for j in range(len(A_in)):
        G1 += PA[j]*A_in[j]
        G2 += PA[j]*A_out[j]

    G = G1/G2

    return kB*T*np.log(G)/e


def Nernst(z,n_in,n_out):
    return kB*T*np.log(n_out/n_in)/(z*e)
    

M_in = [ 0.0001, 150, 15 ]
M_out = [ 2.5, 5, 145]
PM = [ 0.0001, 1, 0.02]
A_in = [10]
A_out = [110]
PA = [0.1]

n_in = [ 0.0001,10,150,15]
n_out = [ 2.5,110,5,145]
z = [2, -1, 1, 1]
name = ['calcium', 'chloride', 'potassium', 'sodium']
Nernst_value = []

for i in range(len(z)):
    Nernst_value.append(Nernst(z[i],n_in[i],n_out[i]))
    print(f"Nernst_{name[i]}= {Nernst_value[i]:.3f}V")


GHK_value = GHK(PM,PA,M_in,M_out,A_in,A_out)
print(f"{'='*20}")
print(f"GHK = {GHK_value:.3f}V")
