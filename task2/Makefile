main: ./src/main.c
	gcc -Wall -std=c99 -o ./bin/main ./src/main.c -lm

main_mpi: ./src/main_mpi.c
	mpicc -Wall -std=c99 -o ./bin/main_mpi ./src/main_mpi.c -lm

main_mpi_openmp: ./src/main_mpi_openmp.c
	mpixlc_r -qsmp=omp -o ./bin/main_mpi_openmp ./src/main_mpi_openmp.c