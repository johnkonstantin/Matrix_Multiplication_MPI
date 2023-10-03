#include <stdio.h>
#include <mpi.h>
#include <malloc.h>

#define P1 2
#define P2 12
#define N1 (500*24)
#define N2 4500
#define N3 (1000*24)


int main(int argc, char** argv) {
	double begin, end;
	MPI_Init(&argc, &argv);
	begin = MPI_Wtime();
	int num_process;
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double* A = NULL;
	double* B = NULL;
	double* C = NULL;
	double* A_p = NULL;
	double* B_p = NULL;
	double* C_p = NULL;

	MPI_Comm gridComm;
	MPI_Comm rowComm;
	MPI_Comm columnComm;
	int coords[2];

	int dimm[2];
	dimm[0] = P1;
	dimm[1] = P2;
	int period[2];
	period[0] = 0;
	period[1] = 0;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimm, period, 0, &gridComm);
	MPI_Comm_rank(gridComm, &rank);
	MPI_Cart_coords(gridComm, rank, 2, coords);

	if (rank == 0) {
		A = malloc(N1 * N2 * sizeof(double));
		B = malloc(N2 * N3 * sizeof(double));
		C = malloc(N1 * N3 * sizeof(double));

		for (int i = 0; i < N1 * N2; ++i) {
			A[i] = i;
		}
		for (int i = 0; i < N2 * N3; ++i) {
			B[i] = i;
		}
	}

	int varCoords[2];
	varCoords[0] = 0;
	varCoords[1] = 1;
	MPI_Cart_sub(gridComm, varCoords, &rowComm);
	varCoords[0] = 1;
	varCoords[1] = 0;
	MPI_Cart_sub(gridComm, varCoords, &columnComm);

	A_p = malloc(N1 * N2 / P1 * sizeof(double));

	if (coords[1] == 0) {
		MPI_Scatter(A, N1 * N2 / P1, MPI_DOUBLE, A_p, N1 * N2 / P1, MPI_DOUBLE, 0, columnComm);
	}

	B_p = malloc(N2 * N3 / P2 * sizeof(double));

	if (coords[0] == 0) {
		MPI_Datatype colType_t, columnsTypeSend, columnsTypeRecv;
		MPI_Type_vector(N2, 1, N3, MPI_DOUBLE, &colType_t);
		MPI_Datatype t;
		MPI_Type_create_resized(colType_t, 0, 1 * sizeof(double), &t);
		MPI_Type_contiguous(N3 / P2, t, &columnsTypeSend);
		MPI_Type_commit(&columnsTypeSend);

		MPI_Type_vector(N2, N3 / P2, N3 / P2, MPI_DOUBLE, &colType_t);
		MPI_Type_create_resized(colType_t, 0, N3 / P2 * sizeof(double), &columnsTypeRecv);
		MPI_Type_commit(&columnsTypeRecv);

		MPI_Scatter(B, 1, columnsTypeSend, B_p, 1, columnsTypeRecv, 0, rowComm);
	}

	MPI_Bcast(A_p, N1 * N2 / P1, MPI_DOUBLE, 0, rowComm);
	MPI_Bcast(B_p, N2 * N3 / P2, MPI_DOUBLE, 0, columnComm);

	C_p = malloc(N1 * N3 / P1 / P2 * sizeof(double));

	for (int i = 0; i < N1 / P1; ++i) {
		for (int j = 0; j < N3 / P2; ++j) {
			C_p[i * N3 / P2 + j] = 0;
			for (int k = 0; k < N2; ++k) {
				C_p[i * N3 / P2 + j] += A_p[i * N2 + k] * B_p[j * N2 + k];
			}
		}
	}

	MPI_Datatype block_t, blockTypeSend, blockTypeRecv;
	MPI_Type_vector(N1 / P1, N3 / P2, N3 / P2, MPI_DOUBLE, &block_t);
	MPI_Type_create_resized(block_t, 0, N3 / P2 * sizeof(double), &blockTypeSend);
	MPI_Type_commit(&blockTypeSend);

	MPI_Type_vector(N1 / P1, N3 / P2, N3, MPI_DOUBLE, &block_t);
	MPI_Type_create_resized(block_t, 0, N3 / P2 * sizeof(double), &blockTypeRecv);
	MPI_Type_commit(&blockTypeRecv);


	int* recvCount = malloc(num_process * sizeof(int));
	int* displs = malloc(num_process * sizeof(int));

	for (int i = 0; i < num_process; ++i) {
		recvCount[i] = 1;
	}

	{
		int disp = 0;
		for (int i = 0; i < P1; ++i) {
			for (int j = 0; j < P2; ++j) {
				displs[i * P2 + j] = disp;
				++disp;
			}
			disp += (N1 / P1 - 1) * P2;
		}
	}


	MPI_Gatherv(C_p, 1, blockTypeSend, C, recvCount, displs, blockTypeRecv, 0, gridComm);
	free(recvCount);
	free(displs);

	end = MPI_Wtime();

	if (rank == 0) {
		printf("C[0] = %f, Time = %f seconds\n", C[0], end - begin);
	}

	free(A);
	free(B);
	free(C);
	free(A_p);
	free(B_p);
	free(C_p);
	MPI_Finalize();
	return 0;
}
