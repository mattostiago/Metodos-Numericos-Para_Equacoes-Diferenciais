import numpy as np
import matplotlib.pyplot as plt


cenarios = [
    {"t_final": 30, "alpha": 2.0, "k": 0.1}
]

Nx = 100
Lx = 30.0
CE = 1.0
dt = 0.01     

plt.figure(figsize=(10, 6))

for idx, cenario in enumerate(cenarios):
    t_final = cenario["t_final"]
    alpha = cenario["alpha"]
    k = cenario["k"]

    dx = Lx / (Nx - 1)            
    Nt = int(t_final / dt)        

    Φ = np.zeros(Nx)              
    Φ_n1 = np.zeros(Nx)           
    Φ[0] = CE                     

    A = np.zeros((Nx, Nx))
    B = np.zeros(Nx)

    for i in range(1, Nx-1):
        A[i, i-1] = -alpha * dt / dx**2
        A[i, i] = 1 + 2 * alpha * dt / dx**2 + k * dt
        A[i, i+1] = -alpha * dt / dx**2

    A[0, 0] = 1                   
    A[Nx-1, Nx-1] = 1 + 2 * alpha * dt / dx**2 
    A[Nx-1, Nx-2] = -2 * alpha * dt / dx**2

    for n in range(1, Nt + 1):
        B[:] = Φ
        B[0] = CE
        B[Nx-1] = Φ[Nx-1]
        Φ_n1 = np.linalg.solve(A, B)  
        Φ = Φ_n1.copy()              

    
    x = np.linspace(0, Lx, Nx)
    plt.plot(x, Φ, label=f'Cenário 1: t={t_final}s, alpha={alpha}, k={k}')

plt.xlabel('Espaço (x)')
plt.ylabel('Concentração (Φ)')
plt.title('Distribuição de Concentração no Domínio do Espaço')
plt.legend()
plt.grid()
plt.show()
