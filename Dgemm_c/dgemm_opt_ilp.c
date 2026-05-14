#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stddef.h>

#define ALINHAR_64B 64
#define PASSO_DESENROLAR (4)

// Executa a multiplicação de matrizes com desenrolamento de laço e vetorização SIMD
void executar_dgemm_v2(size_t dim, double* mat_p1, double* mat_p2, double* mat_saida) {
    // O pulo no laço externo considera o fator de desenrolamento (4 * 4 = 16 elementos por vez)
    for (size_t i = 0; i < dim; i += 4 * PASSO_DESENROLAR) {
        for (size_t j = 0; j < dim; j++) {
            
            __m256d regs_acumuladores[PASSO_DESENROLAR];
            
            // Carga dos registradores com os valores atuais da matriz de saída
            for (int d = 0; d < PASSO_DESENROLAR; d++) {
                regs_acumuladores[d] = _mm256_load_pd(mat_saida + i + (j * dim) + (d * 4));
            }

            for (size_t k = 0; k < dim; k++) {
                // Broadcast do elemento escalar para multiplicação vetorial
                __m256d b_broadcast = _mm256_broadcast_sd(mat_p1 + (j * dim) + k); 
                
                for (int d = 0; d < PASSO_DESENROLAR; d++) {
                    __m256d a_vetor = _mm256_load_pd(mat_p2 + (dim * k) + i + 4 * d);
                    regs_acumuladores[d] = _mm256_add_pd(regs_acumuladores[d], _mm256_mul_pd(b_broadcast, a_vetor));
                }
            }

            // Armazena os resultados processados de volta na memória
            for (int d = 0; d < PASSO_DESENROLAR; d++) {
                _mm256_store_pd(mat_saida + i + (j * dim) + (d * 4), regs_acumuladores[d]);
            }
        }
    }
}

// Preenchimento de dados para teste
void preencher_matriz_rand(size_t n, double* ptr_matriz) {
    for (size_t i = 0; i < n * n; i++) {
        ptr_matriz[i] = (double)rand() / RAND_MAX;
    }
}

// Limpeza do buffer de saída
void inicializar_com_zeros(size_t n, double* ptr_matriz) {
    for (size_t i = 0; i < n * n; i++) {
        ptr_matriz[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Erro: Informe o tamanho N da matriz.\n");
        return 1;
    }

    size_t N = (size_t)atoi(argv[1]);
    // Verificação de alinhamento necessária para o passo de desenrolar
    if (N <= 0 || N % 16 != 0) {
        printf("Erro: N deve ser múltiplo de 16 para esta implementação.\n");
        return 1;
    }

    srand((unsigned int)time(NULL));

    for (int ciclo = 1; ciclo <= 5; ciclo++) {
        // Alocação de memória alinhada para instruções PD
        double* buffer_a = (double*)_mm_malloc(N * N * sizeof(double), ALINHAR_64B);
        double* buffer_b = (double*)_mm_malloc(N * N * sizeof(double), ALINHAR_64B);
        double* buffer_c = (double*)_mm_malloc(N * N * sizeof(double), ALINHAR_64B);

        if (!buffer_a || !buffer_b || !buffer_c) {
            printf("Erro: Memória insuficiente.\n");
            return 1;
        }

        preencher_matriz_rand(N, buffer_a);
        preencher_matriz_rand(N, buffer_b);
        inicializar_com_zeros(N, buffer_c);

        clock_t t_abre = clock();
        executar_dgemm_v2(N, buffer_a, buffer_b, buffer_c);
        clock_t t_fecha = clock();

        double tempo_ms = (double)(t_fecha - t_abre) / CLOCKS_PER_SEC * 1000;
        printf("Rodada %d: %.2f ms\n", ciclo, tempo_ms);

        _mm_free(buffer_a);
        _mm_free(buffer_b);
        _mm_free(buffer_c);
    }

    return 0;
}