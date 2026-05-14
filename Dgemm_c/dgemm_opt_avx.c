#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stddef.h>

#define ALINHAMENTO_MEM 64

// Realiza o cálculo otimizado com vetorização AVX (Advanced Vector Extensions)
void calcular_produto_vetorial(size_t ordem, double* m_a, double* m_b, double* m_res) {
    for (size_t col_b = 0; col_b < ordem; col_b++) {
        for (size_t lin_a = 0; lin_a < ordem; lin_a += 4) {
            // Carrega os dados da matriz de resultado para um registrador de 256 bits
            __m256d reg_res = _mm256_load_pd(m_res + lin_a + col_b * ordem);
            
            for (size_t iter = 0; iter < ordem; iter++) {
                // Multiplicação e soma acumulada (fused multiply-add simulado)
                __m256d reg_a = _mm256_load_pd(m_a + lin_a + iter * ordem);
                __m256d reg_b = _mm256_set1_pd(m_b[iter + col_b * ordem]);
                
                reg_res = _mm256_add_pd(reg_res, _mm256_mul_pd(reg_a, reg_b));
            }
            // Salva o resultado de volta na memória
            _mm256_store_pd(m_res + lin_a + col_b * ordem, reg_res);
        }
    }
}

// Gera valores aleatórios decimais para preencher os buffers
void preencher_aleatorio(size_t dim, double* buffer) {
    for (size_t i = 0; i < dim * dim; i++) {
        buffer[i] = (double)rand() / RAND_MAX;
    }
}

// Inicializa o buffer com zeros
void resetar_buffer(size_t dim, double* buffer) {
    for (size_t i = 0; i < dim * dim; i++) {
        buffer[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Erro: Informe a dimensão da matriz.\n");
        return 1;
    }

    size_t N = (size_t)atoi(argv[1]);
    
    // Verificação de segurança para vetorização (múltiplo de 8 para garantir alinhamento)
    if (N <= 0 || N % 8 != 0) {
        printf("Erro: A dimensão deve ser múltipla de 8 para otimização SIMD.\n");
        return 1;
    }

    // Executa o teste 5 vezes conforme solicitado
    for (int rodada = 1; rodada <= 5; rodada++) {
        
        // Alocação alinhada para evitar falhas de segmentação em instruções PD (Packed Double)
        double* mat_1 = (double*)_mm_malloc(N * N * sizeof(double), ALINHAMENTO_MEM);
        double* mat_2 = (double*)_mm_malloc(N * N * sizeof(double), ALINHAMENTO_MEM);
        double* mat_c = (double*)_mm_malloc(N * N * sizeof(double), ALINHAMENTO_MEM);

        if (!mat_1 || !mat_2 || !mat_c) {
            printf("Erro de memória na rodada %d.\n", rodada);
            return 1;
        }

        srand((unsigned int)time(NULL) + rodada);
        preencher_aleatorio(N, mat_1);
        preencher_aleatorio(N, mat_2);
        resetar_buffer(N, mat_c);

        clock_t t_inicio = clock();
        calcular_produto_vetorial(N, mat_1, mat_2, mat_c);
        clock_t t_fim = clock();

        double milissegundos = (double)(t_fim - t_inicio) / CLOCKS_PER_SEC * 1000;
        printf("[%d/5] Tempo de processamento AVX: %.2f ms\n", rodada, milissegundos);

        _mm_free(mat_1);
        _mm_free(mat_2);
        _mm_free(mat_c);
    }

    return 0;
}