#include <iostream>
#include <cstring>
#include "utilities/OBJLoader.hpp"
#include "utilities/lodepng.h"
#include "rasteriser.hpp"
#include <mpi.h>

int main(int argc, char **argv)
{
    std::string input("../input/sphere.obj");
    std::string output("../output/sphere.png");
    std::string outputWithRank;
    std::string format(".png");
    unsigned int width = 1920;
    unsigned int height = 1080;
    unsigned int depth = 3;

    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    int count = 3; //number of elements in struct
    MPI_Aint offsets3[3] = {0, sizeof(float), 2 * sizeof(float)};
    int blocklengths3[3] = {1, 1, 1};
    MPI_Datatype types3[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Datatype mpi_float3;

    MPI_Type_create_struct(count, blocklengths3, offsets3, types3, &mpi_float3);

    MPI_Type_commit(&mpi_float3);
    {
        MPI_Aint typesize;
        MPI_Type_extent(mpi_float3, &typesize);
    }

    count = 4; //number of elements in struct
    MPI_Aint offsets4[4] = {0, sizeof(float), 2 * sizeof(float), 3 * sizeof(float)};
    int blocklengths4[4] = {1, 1, 1, 1};
    MPI_Datatype types4[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Datatype mpi_float4;

    MPI_Type_create_struct(count, blocklengths4, offsets4, types4, &mpi_float4);

    MPI_Type_commit(&mpi_float4);
    {
        MPI_Aint typesize;
        MPI_Type_extent(mpi_float4, &typesize);
    }


    for (int i = 1; i < argc; i++)
    {
        if (i < argc - 1)
        {
            if (std::strcmp("-i", argv[i]) == 0)
            {
                input = argv[i + 1];
            }
            else if (std::strcmp("-o", argv[i]) == 0)
            {
                char rankString[3];
                std::sprintf(rankString, "%d", rank);
                outputWithRank = rankString;
                output = argv[i + 1] + outputWithRank + format;
            }
            else if (std::strcmp("-w", argv[i]) == 0)
            {
                width = (unsigned int) std::stoul(argv[i + 1]);
            }
            else if (std::strcmp("-h", argv[i]) == 0)
            {
                height = (unsigned int) std::stoul(argv[i + 1]);
            }
            else if (std::strcmp("-d", argv[i]) == 0)
            {
                depth = (int) std::stoul(argv[i + 1]);
            }
        }
    }

    std::cout << "Loading '" << input << "' file... " << std::endl;

    int root = 0;
    unsigned int meshSize;
    std::vector<unsigned int> verticesSize;
    std::vector<unsigned int> normalsSize;
    std::vector<unsigned int> texturesSize;

    // just load the materials
    std::vector<Mesh> meshs = loadWavefront(input, rank, root, false);

    // For each mesh i need the sizes of each vector
    if(rank == root)
    {
        for (unsigned int i = 0; i < meshs.size(); ++i)
        {
            verticesSize.push_back(meshs.at(i).vertices.size());
            normalsSize.push_back(meshs.at(i).normals.size());
            texturesSize.push_back(meshs.at(i).textures.size());
        }
        meshSize = meshs.size();
    }

    // Broadcast the size of mesh vector, I'm not sure we need to do this
    MPI_Bcast(&meshSize, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);

    if(rank != root)
    {
        for (unsigned int i = 0; i < meshs.size(); ++i)
        {
            meshs.at(i).vertices.clear();
            meshs.at(i).normals.clear();
            meshs.at(i).textures.clear();
        }

        verticesSize.resize(meshSize);
        normalsSize.resize(meshSize);
        texturesSize.resize(meshSize);
    }

    // Broadcast just the size of each vector
    MPI_Bcast(&verticesSize[0], meshSize, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&normalsSize[0], meshSize, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&texturesSize[0], meshSize, MPI_UNSIGNED, root, MPI_COMM_WORLD);

    // Reserve the right space for each vector
    if(rank != root)
    {
        for (unsigned int i = 0; i < meshSize; ++i)
        {

            meshs.at(i).vertices.resize(verticesSize.at(i));
            meshs.at(i).normals.resize(normalsSize.at(i));
            meshs.at(i).textures.resize(texturesSize.at(i));
        }
    }

    // Pass the vector
    for (unsigned int i = 0; i < meshSize; ++i)
    {
        MPI_Bcast(&meshs.at(i).vertices[0], verticesSize.at(i),  mpi_float4, root, MPI_COMM_WORLD);
        MPI_Bcast(&meshs.at(i).normals[0], normalsSize.at(i),  mpi_float3, root, MPI_COMM_WORLD);
        MPI_Bcast(&meshs.at(i).textures[0], texturesSize.at(i),  mpi_float3, root, MPI_COMM_WORLD);
    }

    std::vector<unsigned char> frameBuffer = rasterise(meshs, width, height, rank, size, depth);

    std::cout << "Writing image to '" << output << "'..." << std::endl;

    unsigned error = lodepng::encode(output, frameBuffer, width, height);

    if(error)
    {
        std::cout << "An error occurred while writing the image file: " << error << ": " << lodepng_error_text(error) << std::endl;
    }


    MPI_Finalize();

    return 0;
}
