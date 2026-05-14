#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stddef.h>

#define ALINHAR 64 
#define FATOR_DESENROLAR (4)
#define TAM_BLOCO 32

// Processa um sub-bloco da matriz para otimizar o uso do cache L1/L2
void multiplicar_sub_bloco(int total_n, int offset_i, int offset_j, int offset_k, double* m_a, double* m_b, double* m_res) {
    for (int i = offset_i; i < offset_i + TAM_BLOCO; i += FATOR_DESENROLAR * 4) {
        for (int j = offset_j; j < offset_j + TAM_BLOCO; j++) {
            
            __m256d acumuladores[FATOR_DESENROLAR];
            
            // Carregamento inicial dos dados vetorizados
            for (int f = 0; f < FATOR_DESENROLAR; f++) {
                acumuladores[f] = _mm256_load_pd(m_res + (j * total_n) + i + f * 4); 
            }

            for (int k = offset_k; k < offset_k + TAM_BLOCO; k++) {
                // Broadcast do elemento da matriz B para todos os slots do registrador
                __m256d val_b = _mm256_broadcast_sd(m_b + (j * total_n) + k);
                
                for (int f = 0; f < FATOR_DESENROLAR; f++) {
                    __m256d val_a = _mm256_load_pd(m_a + (total_n * k) + i + 4 * f);
                    acumuladores[f] = _mm256_add_pd(acumuladores[f], _mm256_mul_pd(val_a, val_b));
                }
            }

            // Armazenamento dos resultados processados
            for (int f = 0; f < FATOR_DESENROLAR; f++) {
                _mm256_store_pd(m_res + (j * total_n) + i + f * 4, acumuladores[f]);
            }
        }
    }
}

// Gerencia a divisão da matriz em blocos (Tiling)
void dgemm_otimizado(int n, double* m_a, double* m_b, double* m_res) {
    for (int bj = 0; bj < n; bj += TAM_BLOCO) {
        for (int bi = 0; bi < n; bi += TAM_BLOCO) {
            for (int bk = 0; bk < n; bk += TAM_BLOCO) {
                multiplicar_sub_bloco(n, bi, bj, bk, m_a, m_b, m_res);
            }
        }
    }
}

void gerar_matriz_aleatoria(size_t n, double* matriz) {
    for (size_t i = 0; i < n * n; i++) {
        matriz[i] = (double)rand() / RAND_MAX;
    }
}

void zerar_matriz(size_t n, double* matriz) {
    for (size_t i = 0; i < n * n; i++) {
        matriz[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Erro: Forneça a dimensão da matriz.\n");
        return 1;
    }

    size_t dim = (size_t)atoi(argv[1]);
    if (dim <= 0 || dim % 32 != 0) {
        printf("Erro: Para este algoritmo, a dimensão deve ser múltipla de 32 (TAM_BLOCO).\n");
        return 1;
    }

    srand((unsigned int)time(NULL));

    for (int execucao = 1; execucao <= 5; execucao++) {
        double* dados_a = (double*)_mm_malloc(dim * dim * sizeof(double), ALINHAR);
        double* dados_b = (double*)_mm_malloc(dim * dim * sizeof(double), ALINHAR);
        double* dados_c = (double*)_mm_malloc(dim * dim * sizeof(double), ALINHAR);

        if (!dados_a || !dados_b || !dados_c) {
            printf("Falha na alocação de memória.\n");
            return 1;
        }

        gerar_matriz_aleatoria(dim, dados_a);
        gerar_matriz_aleatoria(dim, dados_b);
        zerar_matriz(dim, dados_c);

        clock_t t_inicio = clock();
        dgemm_otimizado((int)dim, dados_a, dados_b, dados_c);
        clock_t t_fim = clock();

        double tempo_total = (double)(t_fim - t_inicio) / CLOCKS_PER_SEC * 1000;
        printf("Teste %d/5: %.2f ms (Dimensão: %zu)\n", execucao, tempo_total, dim);

        _mm_free(dados_a);
        _mm_free(dados_b);
        _mm_free(dados_c);
    }

    return 0;
}