import java.util.Random;

public class CalculoMatriz {

    public static void main(String[] args) {
        int dimensao = 8192; // Defina o tamanho aqui
        double[][] matrizA = criarMatrizAleatoria(dimensao, dimensao);
        double[][] matrizB = criarMatrizAleatoria(dimensao, dimensao);
        double[][] matrizResultado = new double[dimensao][dimensao];

        // Executa o benchmark 5 vezes
        for(int rodada = 0; rodada < 5; rodada++) {
            long tempoInicial = System.nanoTime();
            executarMultiplicacao(dimensao, dimensao, dimensao, matrizA, matrizB, matrizResultado);
            long tempoFinal = System.nanoTime();

            long diferencaTempo = tempoFinal - tempoInicial; 
            double tempoEmMilissegundos = diferencaTempo / 1_000_000.0;

            System.out.printf("Tempo total para dgemm = %.2f ms%n", tempoEmMilissegundos);
        }
    }

    // Cria matriz aleatória com valores entre 0 e 1
    public static double[][] criarMatrizAleatoria(int linhas, int colunas) {
        Random geradorAleatorio = new Random();
        double[][] matrizGerada = new double[linhas][colunas];
        
        for (int i = 0; i < linhas; i++) {
            for (int j = 0; j < colunas; j++) {
                matrizGerada[i][j] = geradorAleatorio.nextDouble();
            }
        }
        return matrizGerada;
    }

    // Realiza a multiplicação de matrizes
    public static void executarMultiplicacao(int m, int n, int k_dim, double[][] a, double[][] b, double[][] c) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double acumulador = 0.0;
                for (int k = 0; k < k_dim; k++) {
                    acumulador += a[i][k] * b[k][j];
                }
                c[i][j] += acumulador;
            }
        }
    }
}