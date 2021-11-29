#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int MAX_ITER = 10000;
double eps = 1e-6;


void* strict_malloc(size_t size) {
    void *new_ptr = malloc(size);
 
    if (new_ptr == NULL) {
        perror("calloc return NULL");
        exit(EXIT_FAILURE);
    }

    return new_ptr;
}


double dot(int M, int N, double h1, double h2, double *x, double *y) {
    double result = 0;
    for (int i = 0; i < (M-1)*(N-1); i++) result += h1 * h2 * x[i] * y[i];
    return result;
}


double k(double x, double y) {
    return 4.0 + x;
}

double q(double x, double y) {
    return (x + y) * (x + y);
}

double u(double x, double y) {
    return exp(1-(x+y)*(x+y));
}

double f(double x, double y) {
    return u(x,y) * ( 6*x + 2*y + 16 - (x+y)*(x+y)*(8*x+31) );
}


void calc_A(int M, int N, double *arr, double *new_arr, double *x, double *y) {
    double arr_ij, arr_i_p_j, arr_i_n_j, arr_i_j_p, arr_i_j_n;
    double h1 = x[1] - x[0];
    double h2 = y[1] - y[0];

    for (int i = 0; i < M-1; i++) {
        for (int j = 0; j < N-1; j++) {
            arr_ij = arr[i*(N-1)+j];
            arr_i_p_j = (i == 0) ? 0 : arr[(i-1)*(N-1)+j];
            arr_i_n_j = (i == M-2) ? 0 : arr[(i+1)*(N-1)+j];
            arr_i_j_p = (j == 0) ? 0 : arr[i*(N-1)+j-1];
            arr_i_j_n = (j == N-2) ? 0 : arr[i*(N-1)+j+1];

            new_arr[i*(N-1)+j] = q(x[i], y[j]) * arr_ij \
                -( k(x[i]+0.5*h1, y[j]) * (arr_i_n_j - arr_ij) - k(x[i]-0.5*h1, y[j]) * (arr_ij - arr_i_p_j) ) / (h1*h1) \
                -( k(x[i], y[j]+0.5*h2) * (arr_i_j_n - arr_ij) - k(x[i], y[j]-0.5*h2) * (arr_ij - arr_i_j_p) ) / (h2*h2);
        }
    }
}


void calc_B(int M, int N, double *arr, double *new_arr, double *x, double *y) {
    double arr_i_p_j, arr_i_n_j, arr_i_j_p, arr_i_j_n;
    double h1 = x[1] - x[0];
    double h2 = y[1] - y[0];

    for (int i = 0; i < M-1; i++) {
        for (int j = 0; j < N-1; j++) {
            arr_i_p_j = i == 0 ? u(x[i]-h1, y[j]) : 0;
            arr_i_n_j = i == M-2 ? u(x[i]+h1, y[j]) : 0;
            arr_i_j_p = j == 0 ? u(x[i], y[j]-h2) : 0;
            arr_i_j_n = j == N-2 ? u(x[i], y[j]+h2) : 0;

            new_arr[i*(N-1)+j] = 
                arr[i*(N-1)+j] \
                - f(x[i], y[j]) \
                - ( k(x[i], y[j]+0.5*h2) * arr_i_j_n + k(x[i], y[j]-0.5*h2) * arr_i_j_p) / (h2*h2) \
                - ( k(x[i]+0.5*h1, y[j]) * arr_i_n_j + k(x[i]-0.5*h1, y[j]) * arr_i_p_j) / (h1*h1); 
        }
    }
}


int solve(int M, int N, double *omega, double *x, double *y, double eps) {
    double *A_r = strict_malloc( (N - 1) * (M - 1) * sizeof(double) );
    double *r = strict_malloc( (N - 1) * (M - 1) * sizeof(double) );
    double tau;
    double h1 = x[1] - x[0];
    double h2 = y[1] - y[0];

    for (int k = 0; k < MAX_ITER; k++) {
        calc_A(M, N, omega, r, x, y);
        calc_B(M, N, r, r, x, y);
        calc_A(M, N, r, A_r, x, y);

        tau = dot(M, N, h1, h2, A_r, r) / dot(M, N, h1, h2, A_r, A_r);

        for (int i = 0; i <  (N - 1) * (M - 1); i++) r[i] = -tau * r[i];

        if (sqrt(dot(M, N, h1, h2, r, r)) < eps) {
            return k;
        }

        for (int i = 0; i <  (N - 1) * (M - 1); i++) omega[i] += r[i];
    }

    free(A_r);
    free(r);

    return MAX_ITER;
}


int main(int argc, char *argv[]) {
    double MIN_X = -1, MAX_X = 2, MIN_Y = -2, MAX_Y = 2;
    int N = 160;
    int M = 160;
    double *x = strict_malloc( (M - 1) * sizeof(double) );
    double *y = strict_malloc( (N - 1) * sizeof(double) );
    double *omega = strict_malloc( (N - 1) * (M - 1) * sizeof(double) );

    double h1 = (MAX_X - MIN_X) / M;
    double h2 = (MAX_Y - MIN_Y) / N;

    // init omega
    for (int i = 0; i < (N - 1) * (M - 1); i++) omega[i] = 1;

    // init x, y
    for (int i = 0; i < (M - 1); i++) x[i] = MIN_X + (i+1)*h1;
    for (int i = 0; i < (N - 1); i++) y[i] = MIN_Y + (i+1)*h2;

    solve(M, N, omega, x, y, eps);

    FILE *fp;
    fp = fopen("./results/consistent_160_160", "w");
    if (fp == NULL) {
        perror("open failed");
        exit(1);
    }

    fprintf(fp, "%d\n", M-1);
    fprintf(fp, "%d", N-1);

    for (int i = 0; i < (M-1)*(N-1); i++)
        fprintf(fp, "\n%.9f", omega[i]);

    fclose(fp);

    free(omega);

    return 0;
}