import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do problema
L = 1.0  # Comprimento da primeira região
L_f = 0.5  # Comprimento da segunda região
D = 1.0  # Coeficiente de difusão
k_a = 1.0  # Parâmetro na região 1
k_b = 2.0  # Parâmetro na região 2
C_E = 1.0  # Condição de contorno em x = 0

# Discretização
N1 = 100  # Número de pontos na primeira região
N2 = 50  # Número de pontos na segunda região
dx1 = L / N1  # Passo de malha na região 1
dx2 = L_f / N2  # Passo de malha na região 2

# Vetor de posições x
x1 = np.linspace(0, L, N1)
x2 = np.linspace(L, L + L_f, N2)
x = np.concatenate((x1, x2))

# Matriz do sistema
N = N1 + N2  # Número total de pontos
A = np.zeros((N, N))
b = np.zeros(N)

# Preencher a matriz A para a região 1 (0 <= x < L)
for i in range(1, N1-1):
    A[i, i-1] = D / dx1**2
    A[i, i] = -2 * D / dx1**2 - k_a
    A[i, i+1] = D / dx1**2

# Condição de contorno C(x=0) = C_E
A[0, 0] = 1
b[0] = C_E

# Preencher a matriz A para a região 2 (L <= x < L + L_f)
for i in range(N1, N-1):
    A[i, i-1] = D / dx2**2
    A[i, i] = -2 * D / dx2**2 - k_b
    A[i, i+1] = D / dx2**2

# Condição de continuidade em x = L
A[N1-1, N1-2] = D / dx1**2
A[N1-1, N1-1] = - (D / dx1**2 + D / dx2**2) - k_a
A[N1-1, N1] = D / dx2**2

# Condição de Neumann (dC/dx = 0) em x = L + L_f
A[-1, -1] = -1 / dx2
A[-1, -2] = 1 / dx2

# Resolver o sistema
C = np.linalg.solve(A, b)

# Plotar o perfil de concentração
plt.plot(x, C, label='Concentração C(x)')
plt.xlabel('x')
plt.ylabel('Concentração C')
plt.title('Perfil de concentração em função de x')
plt.legend()
plt.grid(True)
plt.savefig('concentracao.png')
plt.show()