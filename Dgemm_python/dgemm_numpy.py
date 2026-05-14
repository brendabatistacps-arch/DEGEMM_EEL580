import numpy as np 
import time

# Dimensão da matriz quadrada
tamanho = 400

# Gera matrizes aleatórias com valores entre 0 e 1
matriz_a = np.random.rand(tamanho, tamanho)
matriz_b = np.random.rand(tamanho, tamanho)

print(f"Iniciando benchmark para matriz {tamanho}x{tamanho}...\n")

# Executa 5 rodadas para medição de desempenho
for rodada in range(1, 6):
    momento_inicio = time.time()
    
    # Realiza a multiplicação de matrizes (DGEMM otimizada pelo NumPy)
    matriz_resultado = np.dot(matriz_a, matriz_b)
    
    momento_fim = time.time()

    # Calcula o intervalo de tempo decorrido
    tempo_execucao = momento_fim - momento_inicio

    # Exibe o tempo formatado em segundos
    print(f"Rodada {rodada}: Tempo de processamento = {tempo_execucao:.6f} segundos")

print("\nBenchmark concluído.")