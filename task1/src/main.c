#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

int root = 0;  // number of master process
int N = 10000;  // count points on worker process
int MAX_ITER = 100000;  // max of count iteration


void* strict_malloc(size_t size) {
    void *new_ptr = malloc(size);
 
    if (new_ptr == NULL) {
        perror("calloc return NULL");
        exit(EXIT_FAILURE);
    }

    return new_ptr;
}


void generate_points(int N, double *points) {
    for(int i = 0; i < 3 * N; i++) points[i] = (double)rand()/RAND_MAX;
}


double f(double x, double y, double z) {
    return (x*x + y*y + z*z > 1) ? 0 : sin(x*x + z*z)*y;
}


double integral(int N, double sum) {
    return sum / N;
}


double calc_integral(int N, double* points) {
    double result = 0;
    for(int i = 0; i < N; i++) result +=f(points[3*i], points[3*i+1], points[3*i+2]);
    return result;
}


int main(int argc, char *argv[]) {
    double analytic_solution = M_PI * (1-sin(1)) / 8;

    MPI_Init(&argc, &argv);

    int size = 1, rank = 0, flag_finished = 0;
    double sum = 0, time_exec;
    double eps = atof(argv[1]);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size == 1)
        perror("Count of process = 1, must be > 1\n");

    if (rank == root) {
        time_exec = MPI_Wtime();

        srand(42);  // Intializes random number generator

        int *sendcounts = (int*) strict_malloc(size * sizeof(int));
        int *displs = (int*) strict_malloc(size * sizeof(int));
        double *points = (double*) strict_malloc(3 * N * sizeof(double));
        double error = 0;

        sendcounts[0] = 0;
        displs[0] = 0;
        for (int i = 1; i < size; i++) sendcounts[i] = 3 * ( N / (size-1) + ( (i-1) < (N % (size-1)) ) );
        for (int i = 1; i < size; i++) displs[i] = displs[i-1] + sendcounts[i-1];

        int k = 0;
        for(k = 0; k < MAX_ITER; k++) {
            generate_points(N, points);

            MPI_Scatterv(points, sendcounts, displs, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, root, MPI_COMM_WORLD);

            MPI_Reduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
            error = fabs( analytic_solution - integral(N * (k + 1), sum) );
            flag_finished = fabs(error/analytic_solution) < eps;
            MPI_Bcast(&flag_finished, 1, MPI_INT, root, MPI_COMM_WORLD);
            if (flag_finished) {
                k += 1;
                break;
            }
        }

        time_exec = MPI_Wtime() - time_exec;
        MPI_Reduce(MPI_IN_PLACE, &time_exec, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        free(points);

        printf("%.16f,%.16f,%d,%.16f\n", integral(N * k, sum), error, N*k, time_exec);
    } else {
        time_exec = MPI_Wtime();
        int M = N / (size-1) + ( (rank-1) < (N % (size-1)) );
        double *points = (double*) strict_malloc(3 * M * sizeof(double));

        for(int k = 0; k < MAX_ITER; k++) {
            MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, points, 3 * M, MPI_DOUBLE, root, MPI_COMM_WORLD);
            sum = calc_integral(M, points);
            MPI_Reduce(&sum, NULL, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
            MPI_Bcast(&flag_finished, 1, MPI_INT, root, MPI_COMM_WORLD);
            if (flag_finished) break;
        }

        time_exec = MPI_Wtime() - time_exec;
        MPI_Reduce(&time_exec, NULL, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

        free(points);
    }

    MPI_Finalize();

    return 0;
}
