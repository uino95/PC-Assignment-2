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

    std::vector<Mesh> meshs = loadWavefront(input, false);

    std::vector<unsigned char> frameBuffer = rasterise(meshs, width, height, rank*10, depth);

    std::cout << "Writing image to '" << output << "'..." << std::endl;

    unsigned error = lodepng::encode(output, frameBuffer, width, height);

    if(error)
    {
        std::cout << "An error occurred while writing the image file: " << error << ": " << lodepng_error_text(error) << std::endl;
    }


    MPI_Finalize();

    return 0;
}
