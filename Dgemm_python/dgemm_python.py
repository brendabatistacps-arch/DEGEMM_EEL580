import random
import time

# Define a dimensão da matriz
tamanho_n = 16

# Gera as matrizes com valores aleatórios entre 0 e 1
mat_entrada_a = [[random.random() for _ in range(tamanho_n)] for _ in range(tamanho_n)]
mat_entrada_b = [[random.random() for _ in range(tamanho_n)] for _ in range(tamanho_n)]
mat_calculada = [[0 for _ in range(tamanho_n)] for _ in range(tamanho_n)]

def realizar_calculo_dgemm(m1, m2, m_res):
    # Captura tempos de início
    t_inicio_real, t_inicio_cpu = time.time(), time.process_time()
    
    # Executa a multiplicação das matrizes
    for i in range(tamanho_n):
        for j in range(tamanho_n):
            for k in range(tamanho_n):
                m_res[i][j] += m1[i][k] * m2[k][j]

    # Captura tempos de término
    t_fim_real, t_fim_cpu = time.time(), time.process_time()
    
    duracao_total = t_fim_real - t_inicio_real
    duracao_cpu = t_fim_cpu - t_inicio_cpu
    
    return m_res, duracao_total, duracao_cpu

# Chama a função e desempacota os resultados
resultado_final, tempo_total_gasto, tempo_cpu_gasto = realizar_calculo_dgemm(mat_entrada_a, mat_entrada_b, mat_calculada)

# Exibe os tempos formatados
print("\nTempo decorrido na multiplicação de matrizes: {:.2f} segundos".format(tempo_total_gasto))
print("\nTempo decorrido (CPU) na multiplicação de matrizes: {:.2f} segundos".format(tempo_cpu_gasto))