import numpy as np
import matplotlib.pyplot as plt

# Definição de parâmetros do problema
Lx = 1.0  # Comprimento do domínio
CE = 1.0  # Condição de contorno em x=0

# Valores para teste
alpha_values = [0.1, 0.01]  # Valores de alpha
u_values = [0.5, 1.0]  # Valores de u
nx_values = [20, 40, 80, 160]  # Refinamento de malha no espaço
T_values = [0.1, 0.5, 1.0]  # Tempos finais de simulação

def solve_advection_diffusion(Lx, T, nx, alpha, u, CE):
    """
    Resolve a equação de advecção-difusão utilizando diferenças finitas explícitas.

    Parâmetros:
    - Lx: Comprimento do domínio
    - T: Tempo final de simulação
    - nx: Número de nós no espaço (refinamento de malha)
    - alpha: Coeficiente de difusão
    - u: Velocidade de advecção
    - CE: Condição de contorno em x=0

    Retorna:
    - x: Posição espacial
    - C: Perfil de concentração no tempo final
    """
    dx = Lx / (nx - 1)
    x = np.linspace(0, Lx, nx)

    # Cálculo de dt respeitando a restrição de estabilidade
    dt_max = 1.0 / ((2 * alpha / dx**2) + (u / dx))
    nt = int(np.ceil(T / dt_max))  # Ajusta o número de passos para garantir estabilidade
    dt = T / nt  # Ajuste para coincidir com o tempo total

    # Inicialização da concentração
    C = np.zeros(nx)
    C[0] = CE  # Condição de contorno em x=0

    for n in range(nt):
        Cn = C.copy()
        for i in range(1, nx - 1):
            # Aproximações de diferenças finitas
            dCdx = (Cn[i] - Cn[i - 1]) / dx  # Recuada no espaço
            d2Cdx2 = (Cn[i + 1] - 2 * Cn[i] + Cn[i - 1]) / dx**2  # Centradas no espaço
            # Atualização explícita
            C[i] = Cn[i] - dt * (u * dCdx - alpha * d2Cdx2)
        # Condição de contorno em x=Lx (derivada nula)
        C[-1] = C[-2]

    return x, C

# Resultados de simulação
results = []
for alpha in alpha_values:
    for u in u_values:
        for nx in nx_values:
            for T in T_values:
                print(f"Simulando para alpha={alpha}, u={u}, nx={nx}, T={T}")
                x, C = solve_advection_diffusion(Lx, T, nx, alpha, u, CE)
                results.append({
                    "alpha": alpha,
                    "u": u,
                    "nx": nx,
                    "T": T,
                    "x": x,
                    "C": C
                })

# Apresentação de gráficos
fig, axs = plt.subplots(len(alpha_values), len(u_values), figsize=(12, 8), sharex=True, sharey=True)
fig.suptitle('Perfis de C(x) para diferentes parâmetros')

for idx, result in enumerate(results):
    i = alpha_values.index(result['alpha'])
    j = u_values.index(result['u'])
    ax = axs[i, j]
    label = f"nx={result['nx']}, T={result['T']:.2f}"
    ax.plot(result['x'], result['C'], label=label)
    ax.set_title(f"alpha={result['alpha']}, u={result['u']}")
    ax.set_xlabel('x')
    ax.set_ylabel('C')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# Tabelas numéricas para análise
import pandas as pd

print("\nResultados numéricos (primeiros 10 pontos para cada simulação):")
for result in results:
    df = pd.DataFrame({"x": result["x"], "C": result["C"]})
    print(f"\nalpha={result['alpha']}, u={result['u']}, nx={result['nx']}, T={result['T']:.2f}")
    print(df.head(10))  # Exibe os 10 primeiros pontos
