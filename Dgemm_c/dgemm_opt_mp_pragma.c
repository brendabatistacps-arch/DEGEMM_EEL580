#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stddef.h>
#include <omp.h>

#define ALINHAMENTO 64
#define DESENROLAR_BLOQ (4)
#define TAMANHO_PEDACO 32

// Processa a multiplicação de um bloco específico da matriz utilizando AVX e Unrolling
void processar_bloco_paralelo(int dim_total, int i_inicio, int j_inicio, int k_inicio, double* mat_a, double* mat_b, double* mat_c) {
    for (int i = i_inicio; i < i_inicio + TAMANHO_PEDACO; i += DESENROLAR_BLOQ * 4) {
        for (int j = j_inicio; j < j_inicio + TAMANHO_PEDACO; j++) {
            
            __m256d regs_acumulados[DESENROLAR_BLOQ];
            
            // Carrega os dados atuais do bloco de destino
            for (int r = 0; r < DESENROLAR_BLOQ; r++) {
                regs_acumulados[r] = _mm256_load_pd(mat_c + (j * dim_total) + i + r * 4);
            }

            for (int k = k_inicio; k < k_inicio + TAMANHO_PEDACO; k++) {
                // Broadcast do elemento da matriz B (espalha o valor para o registrador vetorial)
                __m256d b_vetor = _mm256_broadcast_sd(mat_b + (j * dim_total) + k);
                
                for (int r = 0; r < DESENROLAR_BLOQ; r++) {
                    __m256d a_vetor = _mm256_load_pd(mat_a + (dim_total * k) + i + 4 * r);
                    regs_acumulados[r] = _mm256_add_pd(regs_acumulados[r], _mm256_mul_pd(a_vetor, b_vetor));
                }
            }

            // Descarrega os registradores de volta para a memória principal
            for (int r = 0; r < DESENROLAR_BLOQ; r++) {
                _mm256_store_pd(mat_c + (j * dim_total) + i + r * 4, regs_acumulados[r]);
            }
        }
    }
}

// Coordena a divisão da matriz em blocos e distribui entre as threads do processador
void dgemm_multicore(int n, double* mat_a, double* mat_b, double* mat_c) {
    // Paraleliza o laço externo para distribuir blocos entre diferentes núcleos
    #pragma omp parallel for collapse(2)
    for (int sj = 0; sj < n; sj += TAMANHO_PEDACO) {
        for (int si = 0; si < n; si += TAMANHO_PEDACO) {
            for (int sk = 0; sk < n; sk += TAMANHO_PEDACO) {
                processar_bloco_paralelo(n, si, sj, sk, mat_a, mat_b, mat_c);
            }
        }
    }
}

void preencher_matriz(size_t n, double* matriz) {
    for (size_t i = 0; i < n * n; i++) {
        matriz[i] = (double)rand() / RAND_MAX;
    }
}

void limpar_matriz(size_t n, double* matriz) {
    for (size_t i = 0; i < n * n; i++) {
        matriz[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Uso: %s <dimensao>\n", argv[0]);
        return 1;
    }

    size_t N = (size_t)atoi(argv[1]);
    if (N <= 0 || N % 32 != 0) {
        printf("Erro: A dimensão deve ser múltipla de 32 para o particionamento de blocos.\n");
        return 1;
    }

    srand((unsigned int)time(NULL));

    for (int teste = 1; teste <= 5; teste++) {
        double* a_ptr = (double*)_mm_malloc(N * N * sizeof(double), ALINHAMENTO);
        double* b_ptr = (double*)_mm_malloc(N * N * sizeof(double), ALINHAMENTO);
        double* c_ptr = (double*)_mm_malloc(N * N * sizeof(double), ALINHAMENTO);

        if (!a_ptr || !b_ptr || !c_ptr) {
            printf("Falha crítica de memória.\n");
            return 1;
        }

        preencher_matriz(N, a_ptr);
        preencher_matriz(N, b_ptr);
        limpar_matriz(N, c_ptr);

        // Medição de tempo usando a API do OpenMP para maior precisão em código paralelo
        double inicio = omp_get_wtime();
        dgemm_multicore((int)N, a_ptr, b_ptr, c_ptr);
        double fim = omp_get_wtime();

        double tempo_ms = (fim - inicio) * 1000.0;
        printf("Execução %d/5 | Tempo: %.2f ms | Threads: %d\n", teste, tempo_ms, omp_get_max_threads());

        _mm_free(a_ptr);
        _mm_free(b_ptr);
        _mm_free(c_ptr);
    }

    return 0;
}