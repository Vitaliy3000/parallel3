#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>


int MAX_ITER = 10000;
double eps = 1e-6;
int root = 0;


struct Neighbors {
    int rank;
    int left_proc_num;
    int right_proc_num;
    int top_proc_num;
    int bottom_proc_num;
    int coord_n;
    int coord_m;
    int n;
    int m;
    int count_n;
    int count_m;
    double *from_top_neighbor_values;
    double *from_bottom_neighbor_values;
    double *from_left_neighbor_values;
    double *from_right_neighbor_values;
    double *to_top_neighbor_values;
    double *to_bottom_neighbor_values;
    double *to_left_neighbor_values;
    double *to_right_neighbor_values;
};


void* strict_malloc(size_t size) {
    void *new_ptr = malloc(size);
 
    if (new_ptr == NULL) {
        perror("calloc return NULL");
        exit(EXIT_FAILURE);
    }

    return new_ptr;
}


double dot(double h1, double h2, double *x, double *y, struct Neighbors *neighbors) {
    double result = 0;
    int n = neighbors->count_n;
    int m = neighbors->count_m;

    for (int i = 0; i < m*n; i++) result += h1 * h2 * x[i] * y[i];

    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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


void calc_A(double *arr, double *new_arr, double *x, double *y, struct Neighbors *neighbors) {
    double arr_ij, arr_i_p_j, arr_i_n_j, arr_i_j_p, arr_i_j_n;
    double h1 = x[1] - x[0];
    double h2 = y[1] - y[0];
    int n = neighbors->count_n;
    int m = neighbors->count_m;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            arr_ij = arr[i*n+j];

            arr_i_p_j = (i == 0) ? ( (neighbors->coord_m == 0) ? 0 : neighbors->from_top_neighbor_values[j] ) : arr[(i-1)*n+j];
            arr_i_n_j = (i == m-1) ? ( (neighbors->coord_m == neighbors->m - 1 ) ? 0 : neighbors->from_bottom_neighbor_values[j] ) : arr[(i+1)*n+j];
            arr_i_j_p = (j == 0) ? ( (neighbors->coord_n == 0) ? 0 : neighbors->from_left_neighbor_values[i] ) : arr[i*n+j-1];
            arr_i_j_n = (j == n-1) ? ( (neighbors->coord_n == neighbors->n - 1 ) ? 0 : neighbors->from_right_neighbor_values[i] ) : arr[i*n+j+1];


            new_arr[i*n+j] = q(x[i], y[j]) * arr_ij \
                -( k(x[i]+0.5*h1, y[j]) * (arr_i_n_j - arr_ij) - k(x[i]-0.5*h1, y[j]) * (arr_ij - arr_i_p_j) ) / (h1*h1) \
                -( k(x[i], y[j]+0.5*h2) * (arr_i_j_n - arr_ij) - k(x[i], y[j]-0.5*h2) * (arr_ij - arr_i_j_p) ) / (h2*h2);
        }
    }
}


void calc_B(double *arr, double *new_arr, double *x, double *y, struct Neighbors *neighbors) {
    double arr_i_p_j, arr_i_n_j, arr_i_j_p, arr_i_j_n;
    double h1 = x[1] - x[0];
    double h2 = y[1] - y[0];
    int n = neighbors->count_n;
    int m = neighbors->count_m;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            arr_i_p_j = ( (i == 0) && (neighbors->coord_m == 0) ) ? u(x[i]-h1, y[j]) : 0;
            arr_i_n_j = ( (i == m-1) && (neighbors->coord_m == neighbors->m - 1 ) ) ? u(x[i]+h1, y[j]) : 0;
            arr_i_j_p = ( (j == 0) && (neighbors->coord_n == 0) ) ? u(x[i], y[j]-h2) : 0;
            arr_i_j_n = ( (j == n-1) && (neighbors->coord_n == neighbors->n - 1 ) ) ? u(x[i], y[j]+h2) : 0;

            new_arr[i*n+j] = 
                arr[i*n+j] \
                - f(x[i], y[j]) \
                - ( k(x[i], y[j]+0.5*h2) * arr_i_j_n + k(x[i], y[j]-0.5*h2) * arr_i_j_p) / (h2*h2) \
                - ( k(x[i]+0.5*h1, y[j]) * arr_i_n_j + k(x[i]-0.5*h1, y[j]) * arr_i_p_j) / (h1*h1); 
        }
    }
}


void update(double *arr, struct Neighbors *neighbors) {
    int n = neighbors->count_n;
    int m = neighbors->count_m;

    int count = 0, shift = 0;
    count += (neighbors->top_proc_num != -1) ? 1 : 0;
    count += (neighbors->bottom_proc_num != -1) ? 1 : 0;
    count += (neighbors->left_proc_num != -1) ? 1 : 0;
    count += (neighbors->right_proc_num != -1) ? 1 : 0;

    MPI_Request *requests = strict_malloc( 2 * count * sizeof(MPI_Request) );
    MPI_Status *statuses = strict_malloc( 2 * count * sizeof(MPI_Status) );

    if (neighbors->bottom_proc_num != -1) {
        for (int i =0; i < n; i++) neighbors->to_bottom_neighbor_values[i] = arr[(m-1)*n + i];
        MPI_Isend(neighbors->to_bottom_neighbor_values, n, MPI_DOUBLE, neighbors->bottom_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }
    if (neighbors->top_proc_num != -1) {
        for (int i =0; i < n; i++) neighbors->to_top_neighbor_values[i] = arr[i];
        MPI_Isend(neighbors->to_top_neighbor_values, n, MPI_DOUBLE, neighbors->top_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }
    if (neighbors->right_proc_num != -1) {
        for (int i =0; i < m; i++) neighbors->to_right_neighbor_values[i] = arr[i*n + n-1];
        MPI_Isend(neighbors->to_right_neighbor_values, m, MPI_DOUBLE, neighbors->right_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }
    if (neighbors->left_proc_num != -1) {
        for (int i =0; i < m; i++) neighbors->to_left_neighbor_values[i] = arr[i*n];
        MPI_Isend(neighbors->to_left_neighbor_values, m, MPI_DOUBLE, neighbors->left_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }


    if (neighbors->top_proc_num != -1) {
        MPI_Irecv(neighbors->from_top_neighbor_values, n, MPI_DOUBLE, neighbors->top_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }
    if (neighbors->bottom_proc_num != -1) {
        MPI_Irecv(neighbors->from_bottom_neighbor_values, n, MPI_DOUBLE, neighbors->bottom_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }
    if (neighbors->left_proc_num != -1) {
        MPI_Irecv(neighbors->from_left_neighbor_values, m, MPI_DOUBLE, neighbors->left_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }
    if (neighbors->right_proc_num != -1) {
        MPI_Irecv(neighbors->from_right_neighbor_values, m, MPI_DOUBLE, neighbors->right_proc_num, 0, MPI_COMM_WORLD, &requests[shift]);
        shift += 1;
    }

    MPI_Waitall(2 * count, requests, statuses);

    free(requests);
    free(statuses);
}


int solve(double *omega, double *x, double *y, double eps, struct Neighbors *neighbors) {
    int n = neighbors->count_n;
    int m = neighbors->count_m;

    double *A_r = strict_malloc( n * m * sizeof(double) );
    double *r = strict_malloc( n * m * sizeof(double) );
    double tau;
    double h1 = x[1] - x[0];
    double h2 = y[1] - y[0];

    for (int k = 0; k < MAX_ITER; k++) {
        update(omega, neighbors);
        calc_A(omega, r, x, y, neighbors);
        calc_B(r, r, x, y, neighbors);
        update(r, neighbors);
        calc_A(r, A_r, x, y, neighbors);

        tau = dot(h1, h2, A_r, r, neighbors) / dot(h1, h2, A_r, A_r, neighbors);

        for (int i = 0; i < n * m; i++) r[i] = -tau * r[i];

        if (sqrt(dot(h1, h2, r, r, neighbors)) < eps)
            return k;

        for (int i = 0; i <  neighbors->count_n * neighbors->count_m; i++) omega[i] += r[i];
    }

    free(A_r);
    free(r);

    return MAX_ITER;
}


int bigger_or_equal(float x, float y) {
    return ( (x > y) || (fabs(x-y) < 10e-10) ) ? 1 : 0;
}


int find_split(int N, int M, int size) {
    int n, m;
    for (n = size; n > 0; n--) {
        m = size / n;
        if ( bigger_or_equal(2.0 * M / N, (double) m / n) || bigger_or_equal((double) m / n, M / (2.0 * N)) )
            return n;
    }

    return 0;
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    double time_exec = MPI_Wtime();

    double MIN_X = -1, MAX_X = 2, MIN_Y = -2, MAX_Y = 2;
    struct Neighbors neighbors;

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    int size = 1, rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    neighbors.rank = rank;
    neighbors.n = find_split(N-1, M-1, size);

    if (neighbors.n == 0) {
        if (rank == root)
            perror("Count of process not enough\n");

        MPI_Finalize();
        exit(1);
    }

    neighbors.m = size / neighbors.n;
    size = neighbors.n * neighbors.m;

    neighbors.coord_n = rank % neighbors.n;
    neighbors.coord_m = rank / neighbors.n;

    int *sendcounts_n = (int*) strict_malloc(neighbors.n * sizeof(int));
    int *displs_n = (int*) strict_malloc(neighbors.n * sizeof(int));

    displs_n[0] = 0;
    for (int i = 0; i < neighbors.n; i++) sendcounts_n[i] = (N - 1) / neighbors.n + ( i < (N - 1) % neighbors.n );
    for (int i = 1; i < neighbors.n; i++) displs_n[i] = displs_n[i-1] + sendcounts_n[i-1];

    int *sendcounts_m = (int*) strict_malloc(neighbors.m * sizeof(int));
    int *displs_m = (int*) strict_malloc(neighbors.m * sizeof(int));

    displs_m[0] = 0;
    for (int i = 0; i < neighbors.m; i++) sendcounts_m[i] = (M - 1) / neighbors.m + ( i < (M - 1) % neighbors.m );
    for (int i = 1; i < neighbors.m; i++) displs_m[i] = displs_m[i-1] + sendcounts_m[i-1];

    neighbors.count_n = sendcounts_n[neighbors.coord_n];
    neighbors.count_m = sendcounts_m[neighbors.coord_m];

    if (neighbors.coord_m == 0) {
        neighbors.top_proc_num = -1;
    } else {
        neighbors.top_proc_num = (neighbors.coord_m-1)*neighbors.n + neighbors.coord_n;
        neighbors.to_top_neighbor_values = (double*) strict_malloc(neighbors.count_n * sizeof(double));
        neighbors.from_top_neighbor_values = (double*) strict_malloc(neighbors.count_n * sizeof(double));
    }

    if (neighbors.coord_n == 0) {
        neighbors.left_proc_num = -1;
    } else {
        neighbors.left_proc_num = neighbors.coord_m*neighbors.n + neighbors.coord_n - 1;
        neighbors.to_left_neighbor_values = (double*) strict_malloc(neighbors.count_m * sizeof(double));
        neighbors.from_left_neighbor_values = (double*) strict_malloc(neighbors.count_m * sizeof(double));
    }

    if (neighbors.coord_n == neighbors.n-1) {
        neighbors.right_proc_num = -1;
    } else {
        neighbors.right_proc_num = neighbors.coord_m*neighbors.n + neighbors.coord_n + 1;
        neighbors.to_right_neighbor_values = (double*) strict_malloc(neighbors.count_m * sizeof(double));
        neighbors.from_right_neighbor_values = (double*) strict_malloc(neighbors.count_m * sizeof(double));
    }

    if (neighbors.coord_m == neighbors.m-1) {
        neighbors.bottom_proc_num = -1;
    } else {
        neighbors.bottom_proc_num = (neighbors.coord_m+1)*neighbors.n + neighbors.coord_n;
        neighbors.to_bottom_neighbor_values = (double*) strict_malloc(neighbors.count_n * sizeof(double));
        neighbors.from_bottom_neighbor_values = (double*) strict_malloc(neighbors.count_n * sizeof(double));
    }

    double *x = strict_malloc( neighbors.count_m * sizeof(double) );
    double *y = strict_malloc( neighbors.count_n * sizeof(double) );
    double *omega = strict_malloc( neighbors.count_n * neighbors.count_m * sizeof(double) );

    double h1 = (MAX_X - MIN_X) / M;
    double h2 = (MAX_Y - MIN_Y) / N;

    // init omega
    for (int i = 0; i < neighbors.count_m * neighbors.count_n; i++) omega[i] = 1;

    // init x, y
    for (int i = 0; i < neighbors.count_m; i++) x[i] = MIN_X + (displs_m[neighbors.coord_m]+1+i)*h1;
    for (int i = 0; i < neighbors.count_n; i++) y[i] = MIN_Y + (displs_n[neighbors.coord_n]+1+i)*h2;

    solve(omega, x, y, eps, &neighbors);

    // double *result_omega;
    // if (rank == root)
    //     result_omega = strict_malloc( (N-1)*(M-1) * sizeof(double) );

    // int shift = 0;
    // for (int i = 0; i < neighbors.m; i++) {
    //     for (int j = 0; j < neighbors.count_m; j++) {
    //         if (rank == root) {
    //             if (neighbors.coord_m == i)
    //                 MPI_Gatherv(&omega[j*neighbors.count_n], neighbors.count_n, MPI_DOUBLE, &result_omega[shift*(N-1)], sendcounts_n, displs_n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    //             else
    //                 MPI_Gatherv(NULL, 0, MPI_DOUBLE, &result_omega[shift*(N-1)], sendcounts_n, displs_n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    //         } else if (neighbors.coord_m == i) {
    //             MPI_Gatherv(&omega[j*neighbors.count_n], neighbors.count_n, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, root, MPI_COMM_WORLD);
    //         }
    //         shift++;
    //     }
    // }

    // if (rank == root) {
    //     FILE *fp;
    //     fp = fopen("./results/parallel_mpi_160_160_1", "w");
    //     if (fp == NULL) {
    //         perror("open failed");
    //         exit(1);
    //     }

    //     fprintf(fp, "%d\n", M-1);
    //     fprintf(fp, "%d", N-1);

    //     for (int i = 0; i < (M-1)*(N-1); i++)
    //         fprintf(fp, "\n%.9f", result_omega[i]);

    //     fclose(fp);
    // }

    free(sendcounts_n);
    free(displs_n);
    free(sendcounts_m);
    free(displs_m);

    if (neighbors.top_proc_num != -1) {
        free(neighbors.to_top_neighbor_values);
        free(neighbors.from_top_neighbor_values);
    }

    if (neighbors.left_proc_num != -1) {
        free(neighbors.to_left_neighbor_values);
        free(neighbors.from_left_neighbor_values);
    }

    if (neighbors.right_proc_num != -1) {
        free(neighbors.to_right_neighbor_values);
        free(neighbors.from_right_neighbor_values);
    }

    if (neighbors.bottom_proc_num != -1) {
        free(neighbors.to_bottom_neighbor_values);
        free(neighbors.from_bottom_neighbor_values);
    }


    free(x);
    free(y);
    free(omega);

    // if (rank == root)
    //     free(result_omega);

    time_exec = MPI_Wtime() - time_exec;

    if (rank == root) {
        MPI_Reduce(MPI_IN_PLACE, &time_exec, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
        printf("%.5f\n", time_exec);
    } else {
        MPI_Reduce(&time_exec, NULL, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
