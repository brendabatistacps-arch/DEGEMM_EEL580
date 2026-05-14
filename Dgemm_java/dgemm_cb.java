import java.util.Random;

public class MatrizCacheBlocking {

    public static void main(String[] args) {
        int tam = 8192; // Defina o tamanho aqui
        double[][] dadosA = gerarMatrizAleatoria(tam, tam);
        double[][] dadosB = gerarMatrizAleatoria(tam, tam);
        double[][] dadosC = new double[tam][tam]; // Matriz resultante

        int tamBloco = 32;

        // Executa o benchmark 5 vezes
        for(int rodada = 0; rodada < 5; rodada++) {
            long cronometroInicio = System.nanoTime();
            executarDgemmCB(tam, tam, tam, dadosA, dadosB, dadosC, tamBloco);
            long cronometroFim = System.nanoTime();
    
            long tempoDecorrido = cronometroFim - cronometroInicio; 
            double tempoEmMs = tempoDecorrido / 1_000_000.0;
    
            System.out.printf("Tempo total para dgemm = %.2f ms%n", tempoEmMs);
        }
    }

    public static double[][] gerarMatrizAleatoria(int lin, int col) {
        Random rand = new Random();
        double[][] matriz = new double[lin][col];
        
        for (int i = 0; i < lin; i++) {
            for (int j = 0; j < col; j++) {
                matriz[i][j] = rand.nextDouble(); 
            }
        }
        return matriz;
    }

    public static void executarDgemmCB(int m, int n, int k_total, double[][] matA, double[][] matB, double[][] matC, int larguraBloco) {

        // dgemm com particionamento de cache (cache blocking)
        for (int bloco_i = 0; bloco_i < m; bloco_i += larguraBloco) {
            for (int bloco_j = 0; bloco_j < n; bloco_j += larguraBloco) {
                for (int bloco_k = 0; bloco_k < k_total; bloco_k += larguraBloco) {
                    
                    // Processamento interno dos blocos
                    for (int i = bloco_i; i < Math.min(bloco_i + larguraBloco, m); i++) {
                        for (int j = bloco_j; j < Math.min(bloco_j + larguraBloco, n); j++) {
                            double acumulador = 0.0;
                            for (int k = bloco_k; k < Math.min(bloco_k + larguraBloco, k_total); k++) {
                                acumulador += matA[i][k] * matB[k][j];
                            }
                            matC[i][j] += acumulador;
                        }
                    }
                }
            }
        }
    }
}