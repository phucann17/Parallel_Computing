#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void mul_naive(int** A, int** B, int** res, unsigned int n) {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int tmp = 0;
            for (int k = 0; k < n; ++k) {
                tmp += A[i][k] * B[k][j];
            }
            res[i][j] = tmp;
        } 
    }
}

void write_log(const char *filename, int** m, unsigned int n){
    FILE* fd = fopen(filename, "w");

    if (!fd){
        perror("Can not open file to write");
        exit(1);
    }

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            fprintf(fd, "%d ", m[i][j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}

int** alloc_matrix(unsigned int n){
    int** m = (int**)malloc(n * sizeof(int*));

    if (!m){
        perror("Can not allocate memory for m");
        exit(1);
    }

    for (int i = 0; i < n; ++i){
        m[i] = (int*)malloc(n * sizeof(int));
        if (!m[i]){
            perror("Can not allocate memory for row of m");
            exit(1);
        }
    }

    return m;
}

void free_matrix(int** m, unsigned int n){
    for (int i = 0; i < n; ++i){
        free(m[i]);
    }
    free(m);
}

int main(){
    unsigned int n = 2000;
    int** A = alloc_matrix(n);
    int** B = alloc_matrix(n);
    int** res = alloc_matrix(n);
    
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            A[i][j] = (int)rand() / (16384*30);
            B[i][j] = (int)rand() / (16384*30);
        }
    }

    double start = omp_get_wtime();
    mul_naive(A, B, res, n);
    double end = omp_get_wtime();

    printf("Time of Naive = %.6f s\n", end - start);
    // write_log("A.txt", A, n);
    // write_log("B.txt", B, n);
    // write_log("res.txt", res, n);  

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(res, n);

}