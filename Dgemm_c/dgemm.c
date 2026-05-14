#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// Realiza a multiplicação de matrizes (Double-precision General Matrix Multiplication)
void calcular_dgemm(size_t dimensao, double* mat_a, double* mat_b, double* mat_res) {
    for (size_t linha = 0; linha < dimensao; linha++) {
        for (size_t col = 0; col < dimensao; col++) {
            for (size_t k = 0; k < dimensao; k++) {
                // Acessa os elementos usando aritmética de ponteiros para matrizes lineares
                mat_res[(linha * dimensao) + col] += mat_a[(linha * dimensao) + k] * mat_b[(k * dimensao) + col];
            }
        }
    }
}

// Preenche a matriz com valores aleatórios entre 0 e 1
void gerar_dados_aleatorios(size_t tam, double* matriz) {
    for (size_t i = 0; i < tam * tam; i++) {
        matriz[i] = (double)rand() / RAND_MAX; 
    }
}

// Zera todos os elementos da matriz resultante
void limpar_matriz(size_t tam, double* matriz) {
    for (size_t i = 0; i < tam * tam; i++) {
        matriz[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Uso: %s <tamanho_da_matriz>\n", argv[0]);
        return 1;
    }

    size_t N = (size_t)atoi(argv[1]);
    if (N <= 0) {
        printf("Erro: O tamanho da matriz deve ser maior que zero.\n");
        return 1;
    }

    // Alocação dinâmica de memória
    double* dados_a = (double*)malloc(N * N * sizeof(double));
    double* dados_b = (double*)malloc(N * N * sizeof(double));
    double* resultado = (double*)malloc(N * N * sizeof(double));

    if (!dados_a || !dados_b || !resultado) {
        printf("Erro Crítico: Memória insuficiente.\n");
        return 1;
    }

    // Inicialização do gerador de números aleatórios e das matrizes
    srand((unsigned int)time(NULL));
    gerar_dados_aleatorios(N, dados_a);
    gerar_dados_aleatorios(N, dados_b);
    limpar_matriz(N, resultado);

    // Medição de performance
    clock_t t_inicio = clock();
    calcular_dgemm(N, dados_a, dados_b, resultado);
    clock_t t_fim = clock();

    // Cálculo do tempo decorrido em milissegundos
    double tempo_ms = (double)(t_fim - t_inicio) / CLOCKS_PER_SEC * 1000;
    
    printf("Tamanho da Matriz: %zu x %zu\n", N, N);
    printf("Tempo de execução: %.2f ms\n", tempo_ms);

    // Liberação de recursos
    free(dados_a);
    free(dados_b);
    free(resultado);

    return 0;
}