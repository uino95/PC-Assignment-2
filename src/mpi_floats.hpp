#pragma once

#include <mpi.h>

int count = 3; //number of elements in struct
MPI_Aint offsets[3] = {0, size_of(float), 2*size_of(float)};
int blocklengths[3] = {1, 1, 1};
MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
MPI_Datatype mpi_float3;

MPI_Type_create_struct(count, blocklengths, offsets, types, &mpi_float3);

count = 4; //number of elements in struct
offsets[4] = {0, size_of(float), 2*size_of(float), 3*size_of(float)};
blocklengths[4] = {1, 1, 1, 1};
types[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
MPI_Datatype mpi_float4;

MPI_Type_create_struct(count, blocklengths, offsets, types, &mpi_float4);