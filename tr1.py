import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Parâmetros físicos
m = 2
c_values = [0, 4]
k_values = [[0.25, 0.5, 0.75, 1.0], [1.0, 10.0, 50.0, 150.0]]
incremento_tempo = 0.1
tempo_final = [15, 10]

# Funções para as equações diferenciais
def dydt(t, y, v, c, k):
    return v

def dvdt(t, y, v, c, k):
    return -(c/m)*v - (k/m)*y

# Método de Runge-Kutta de Quarta Ordem
def runge_kutta(y0, v0, c, k, tempo_final, incremento_tempo):
    num_pontos = int(tempo_final / incremento_tempo) + 1
    tempo = np.linspace(0, tempo_final, num_pontos)
    y = np.zeros(num_pontos)
    v = np.zeros(num_pontos)

    y[0] = y0
    v[0] = v0

    for i in range(1, num_pontos):
        k1y = dydt(tempo[i-1], y[i-1], v[i-1], c, k)
        k1v = dvdt(tempo[i-1], y[i-1], v[i-1], c, k)
        
        k2y = dydt(tempo[i-1] + incremento_tempo/2, y[i-1] + k1y*incremento_tempo/2, v[i-1] + k1v*incremento_tempo/2, c, k)
        k2v = dvdt(tempo[i-1] + incremento_tempo/2, y[i-1] + k1y*incremento_tempo/2, v[i-1] + k1v*incremento_tempo/2, c, k)
        
        k3y = dydt(tempo[i-1] + incremento_tempo/2, y[i-1] + k2y*incremento_tempo/2, v[i-1] + k2v*incremento_tempo/2, c, k)
        k3v = dvdt(tempo[i-1] + incremento_tempo/2, y[i-1] + k2y*incremento_tempo/2, v[i-1] + k2v*incremento_tempo/2, c, k)
        
        k4y = dydt(tempo[i-1] + incremento_tempo, y[i-1] + k3y*incremento_tempo, v[i-1] + k3v*incremento_tempo, c, k)
        k4v = dvdt(tempo[i-1] + incremento_tempo, y[i-1] + k3y*incremento_tempo, v[i-1] + k3v*incremento_tempo, c, k)
        
        y[i] = y[i-1] + (incremento_tempo/6)*(k1y + 2*k2y + 2*k3y + k4y)
        v[i] = v[i-1] + (incremento_tempo/6)*(k1v + 2*k2v + 2*k3v + k4v)

    return tempo, y, v

# Calculando e plotando as soluções numéricas
for c, k_vals, tempo_final in zip(c_values, k_values, tempo_final):
    for k in k_vals:
        y0 = 0.5
        v0 = 0
        tempo, y, v = runge_kutta(y0, v0, c, k, tempo_final, incremento_tempo)
        
        argumento_raiz = k/m - (c**2)/(4*m)
        if argumento_raiz > 0:
            plt.figure()
            plt.plot(tempo, y, label='y(t) numerico')
            plt.plot(tempo, y0*np.exp(-(c/(2*m))*tempo)*np.cos(np.sqrt(argumento_raiz)*tempo), '--', label='y(t) analítico')
            plt.xlabel('Tempo')
            plt.ylabel('Posição y')
            plt.title(f'Solução numérica e analítica para k={k}, c={c}')
            plt.legend()
            plt.show()

            # Criando a tabela como imagem
            table_data = [['Tempo', 'yexato', 'ynumerico', 'vexato', 'vnumerico']]
            for i in range(len(tempo)):
                y_exato = y0*np.exp(-(c/(2*m))*tempo[i])*np.cos(np.sqrt(argumento_raiz)*tempo[i])
                v_exato = -y0*np.exp(-(c/(2*m))*tempo[i])*((c**2 + 4*m*k)/(4*m))*np.sin(np.sqrt(argumento_raiz)*tempo[i])
                table_data.append([f'{tempo[i]:.2f}', f'{y_exato:.4f}', f'{y[i]:.4f}', f'{v_exato:.4f}', f'{v[i]:.4f}'])

            # Criando a imagem da tabela
            cell_width = 120
            cell_height = 30
            img_width = cell_width * len(table_data[0])
            img_height = cell_height * (len(table_data) + 1)
            img = Image.new('RGB', (img_width, img_height), color = (255, 255, 255))
            d = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            for i in range(len(table_data)):
                for j in range(len(table_data[i])):
                    d.text((j*cell_width + 10, i*cell_height + 10), table_data[i][j], font=font, fill=(0, 0, 0))

            img.show()
