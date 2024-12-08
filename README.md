# Métodos Numéricos para Equações Diferenciais I

Este projeto contém implementações de métodos numéricos para resolver equações diferenciais, desenvolvido como parte da disciplina de Métodos Numéricos para Equações Diferenciais I na UERJ.

## Estrutura do Projeto

O projeto está organizado em diferentes arquivos Python, cada um abordando um problema específico:

- `tr1.py`: Solução numérica e analítica de uma equação diferencial de segunda ordem usando o método de Runge-Kutta de quarta ordem.
- `tr2.py`: Solução de um problema de difusão em duas regiões com diferentes parâmetros usando o método de diferenças finitas.
- `tr3.py`: Simulação de um problema de difusão com condições de contorno específicas usando o método de diferenças finitas implícitas.
- `tr4.py`: Resolução da equação de advecção-difusão utilizando diferenças finitas explícitas.

## Requisitos

- Python 3.x
- Bibliotecas: `numpy`, `matplotlib`, `pandas`, `PIL`

Você pode instalar as bibliotecas necessárias usando o seguinte comando:

```bash
pip install numpy matplotlib pandas pillow
```

### Como Executar

```bash
python tr1.py
```
Este script resolve uma equação diferencial de segunda ordem usando o método de Runge-Kutta de quarta ordem e compara a solução numérica com a solução analítica.

```bash
python tr2.py
```
Este script resolve um problema de difusão em duas regiões com diferentes parâmetros usando o método de diferenças finitas.

```bash
python tr3.py
```
Este script simula um problema de difusão com condições de contorno específicas usando o método de diferenças finitas implícitas.

```bash
python tr4.py
```
Este script resolve a equação de advecção-difusão utilizando diferenças finitas explícitas e gera gráficos dos perfis de concentração.

## Resultados
Os resultados das simulações são apresentados em gráficos gerados pelo matplotlib e, em alguns casos, tabelas numéricas são exibidas para análise.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.