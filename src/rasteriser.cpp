#include "rasteriser.hpp"
#include "utilities/lodepng.h"
#include <vector>
#include <iomanip>
#include <chrono>
#include <limits>
#include <mpi.h>
#include <math.h>

const std::vector<globalLight> lightSources = { {{0.3f, 0.5f, 1.0f}, {1.0f, 1.0f, 1.0f}} };

typedef struct perfCounter
{
    unsigned long meshs = 0;
    unsigned long triagnles = 0;
} perfCounter;

perfCounter counter = {};

void runVertexShader( Mesh &mesh,
                      Mesh &transformedMesh,
                      float3 positionOffset,
                      float scale,
                      unsigned int const width,
                      unsigned int const height,
                      float const rotationAngle = 0)
{
    float const pi = std::acos(-1);
    // The matrices defined below are the ones used to transform the vertices and normals.

    // This projection matrix assumes a 16:9 aspect ratio, and an field of view (FOV) of 90 degrees.
    mat4x4 const projectionMatrix(
        0.347270,   0,          0,      0,
        0,          0.617370,   0,      0,
        0,          0,          -1,     -0.2f,
        0,          0,          -1,     0);

    mat4x4 translationMatrix(
        1,          0,          0,          0 + positionOffset.x /*X*/,
        0,          1,          0,          0 + positionOffset.y /*Y*/,
        0,          0,          1,          -10 + positionOffset.z /*Z*/,
        0,          0,          0,          1);

    mat4x4 scaleMatrix(
        scale/*X*/, 0,          0,              0,
        0,          scale/*Y*/, 0,              0,
        0,          0,          scale/*Z*/,     0,
        0,          0,          0,              1);

    mat4x4 const rotationMatrixX(
        1,          0,              0,              0,
        0,          std::cos(0),    -std::sin(0),   0,
        0,          std::sin(0),    std::cos(0),    0,
        0,          0,              0,              1);

    float const rotationAngleRad = (pi / 4.0f) + (rotationAngle / (180.0f / pi));

    mat4x4 const rotationMatrixY(
        std::cos(rotationAngleRad),     0,          std::sin(rotationAngleRad),     0,
        0,                              1,          0,                              0,
        -std::sin(rotationAngleRad),    0,          std::cos(rotationAngleRad),     0,
        0,                              0,          0,                              1);

    mat4x4 const rotationMatrixZ(
        std::cos(pi),   -std::sin(pi),  0,          0,
        std::sin(pi),   std::cos(pi),   0,          0,
        0,              0,              1,          0,
        0,              0,              0,          1);

    mat4x4 const MVP =
        projectionMatrix * translationMatrix * rotationMatrixX * rotationMatrixY * rotationMatrixZ * scaleMatrix;

    for (unsigned int i = 0; i < mesh.vertices.size(); i++)
    {
        float4 currentVertex = mesh.vertices.at(i);
        float4 transformed = (MVP * currentVertex);
        currentVertex = transformed / transformed.w;
        currentVertex.x = (currentVertex.x + 0.5f) * (float) width;
        currentVertex.y = (currentVertex.y + 0.5f) * (float) height;
        transformedMesh.vertices.at(i) = currentVertex;
    }
}


void runFragmentShader( std::vector<unsigned char> &frameBuffer,
                        unsigned int const baseIndex,
                        Face const &face,
                        float3 const &weights )
{
    float3 normal = face.getNormal(weights);

    float3 colour(0);
    for (globalLight const &l : lightSources)
    {
        float3 lightNormal = normal * l.direction;
        colour += (face.parent.material.Kd * l.colour) * (lightNormal.x + lightNormal.y + lightNormal.z);
    }

    colour = colour.clamp(0.0f, 1.0f);
    frameBuffer.at(4 * baseIndex + 0) = colour.x * 255.0f;
    frameBuffer.at(4 * baseIndex + 1) = colour.y * 255.0f;
    frameBuffer.at(4 * baseIndex + 2) = colour.z * 255.0f;
    frameBuffer.at(4 * baseIndex + 3) = 255;
}

/**
 * The main procedure which rasterises all triangles on the framebuffer
 * @param transformedMesh         Transformed mesh object
 * @param frameBuffer             frame buffer for the rendered image
 * @param depthBuffer             depth buffer for every pixel on the image
 * @param width                   width of the image
 * @param height                  height of the image
 */
void rasteriseTriangles( Mesh &transformedMesh,
                         std::vector<unsigned char> &frameBuffer,
                         std::vector<float> &depthBuffer,
                         unsigned int const width,
                         unsigned int const height )
{
    for (unsigned int i = 0; i < transformedMesh.faceCount(); i++)
    {

        Face face = transformedMesh.getFace(i);
        unsigned int minx = int(std::floor(std::min(std::min(face.v0.x, face.v1.x), face.v2.x)));
        unsigned int maxx = int(std::ceil (std::max(std::max(face.v0.x, face.v1.x), face.v2.x)));
        unsigned int miny = int(std::floor(std::min(std::min(face.v0.y, face.v1.y), face.v2.y)));
        unsigned int maxy = int(std::ceil (std::max(std::max(face.v0.y, face.v1.y), face.v2.y)));

        // Let's make sure the screen coordinates stay inside the window
        minx = std::max(minx, (unsigned int) 0);
        maxx = std::min(maxx, width);
        miny = std::max(miny, (unsigned int) 0);
        maxy = std::min(maxy, height);

        // We iterate over each pixel in the triangle's bounding box
        for(unsigned int x = minx; x < maxx; x++)
        {
            for(unsigned int y = miny; y < maxy; y++)
            {
                float u, v, w;
                if(face.inRange(x, y, u, v, w))
                {
                    float pixelDepth = face.getDepth(u, v, w);
                    if( pixelDepth >= -1 && pixelDepth <= 1 && pixelDepth < depthBuffer.at(y * width + x))
                    {
                        depthBuffer.at(y * width + x) = pixelDepth;
                        runFragmentShader(frameBuffer, x + (width * y), face, float3(u, v, w));
                    }
                }
            }
        }
    }
}

/**
* Updates the vector of offsets so to provide the vertices for the current depth.
* That is possible thanks to the scale coefficient, that is correctly passed by the
* renderMeshFractal. Higher scale provides more in-depth image.
* Considering currentOffsets, the newOffsets vector is updated by scaling
* by the "scale" factor.
*/
void updateList(std::vector<float3> &currentOffsets, float largestBoundingBoxSide, float scale, std::vector<float3> &newOffsets)
{

    for (unsigned int i = 0; i < currentOffsets.size(); ++i)
    {
        for(int offsetX = -1; offsetX <= 1; offsetX++)
        {
            for(int offsetY = -1; offsetY <= 1; offsetY++)
            {
                for(int offsetZ = -1; offsetZ <= 1; offsetZ++)
                {
                    float3 offset(offsetX, offsetY, offsetZ);
                    if(offset == 0) {
                      continue;
                    }
                    float3 displacedOffset(currentOffsets.at(i) + offset * (largestBoundingBoxSide / 2.0f) * scale);
                    newOffsets.push_back(displacedOffset);
                }
            }
        }
    }
}

void renderMeshFractal(
    std::vector<Mesh> &meshes,
    std::vector<Mesh> &transformedMeshes,
    unsigned int width,
    unsigned int height,
    std::vector<unsigned char> &frameBuffer,
    std::vector<float> &depthBuffer,
    float largestBoundingBoxSide,
    int rank,
    int size,
    int depthLimit,
    float scale = 1.0,
    float3 distanceOffset = {0, 0, 0})
{
  unsigned int i = 0;
  int currentDepth = 1;
  int limit = 1;
  /**
  * currentOffsets are the overall offsets. We always build all of them,
  * and then we rearrange them into the partials.
  * At first, we initialize it with only the distanceOffset, as the first depth
  * contains only one image
  */
  std::vector<float3> currentOffsets;
  currentOffsets.push_back(distanceOffset);
  /**
  * tmpOffsets is used to build the offsets for the next level of depth.
  * It is necessary as currentOffsets is necessary while doing that, so we need
  * a temporary vector
  */
  std::vector<float3> tmpOffsets;
  /**
  * partialCurrentOffsets instead contains only the offsets that are used
  * by the current process. Same reasoning as for currentOffsets
  * for the representation of the
  */
  std::vector<float3> partialCurrentOffsets;
  partialCurrentOffsets.push_back(distanceOffset);
  /**
  * Once we have the currentOffsets vector, we just need to redistribuite them
  * over the various ranks. To do it in a proper way, we use partialCurrentOffsets,
  * and fill it only if the condition: rank == (k % size) is true (being size the number of processes)
  */
  do {
    /**
    * First branch: representation for the current depth. This branch is taken
    * more times for each branch, until every offset in partialCurrentOffsets
    * is executed.
    */
    if(i < limit)
    {
        /**
        * Drawing objects for the current iteration.
        */
        for (unsigned int j = 0; j < meshes.size(); j++)
        {
            Mesh &mesh = meshes.at(j);
            Mesh &transformedMesh = transformedMeshes.at(j);
            runVertexShader(mesh, transformedMesh, partialCurrentOffsets.at(i), scale, width, height);
            rasteriseTriangles(transformedMesh, frameBuffer, depthBuffer, width, height);
        }
    }
    /**
    * We compute the new offsets by calling updateList only if the next depth
    * is required by the user. Note that this branch contains the parallelization,
    * and that it is never entered in case of DEPTH=1. In that case, we don't
    * have any kind of parallelization.
    */
    else if(currentDepth + 1 < depthLimit)
    {
        // Now we update the list of the offset in a smaller size
        updateList(currentOffsets, largestBoundingBoxSide, scale, tmpOffsets);
        currentOffsets = tmpOffsets;
        tmpOffsets.clear();
        /**
        * Here is where the real parallelization happens.
        * In order to obtain a fair redistribution, we consider the current index,
        * divide it by the number of processes (i.e., size variable) and update
        * partialCurrentOffsets only in case rest is 0
        */
        for (int k = 0; k < currentOffsets.size(); k++) {
          if(rank == k % size){
            partialCurrentOffsets.push_back(currentOffsets.at(k));
          }
        }
        currentDepth++;
        scale = scale / 3.0;
        i = -1;
        limit = partialCurrentOffsets.size();
    }
    /**
    * Otherwise, we have reached the limit and we exit the loop.
    */
    else {
        return;
    }
    i++;
  } while(currentDepth != depthLimit);
}

// This function kicks off the rasterisation process.
std::vector<unsigned char> rasterise(std::vector<Mesh> &meshes, unsigned int width, unsigned int height, int rank, int size, unsigned int depthLimit)
{
    // We first need to allocate some buffers.
    // The framebuffer contains the image being rendered.
    std::vector<unsigned char> frameBuffer;
    // The depth buffer is used to make sure that objects closer to the camera occlude/obscure objects that are behind it
    std::vector<float> depthBuffer;
    frameBuffer.resize(width * height * 4, 0);
    for (unsigned int i = 3; i < (4 * width * height); i += 4)
    {
        frameBuffer.at(i) = 255;
    }
    depthBuffer.resize(width * height, 1);

    float3 boundingBoxMin(std::numeric_limits<float>::max());
    float3 boundingBoxMax(std::numeric_limits<float>::min());

    std::vector<Mesh> transformedMeshes;
    for(unsigned int i = 0; i < meshes.size(); i++)
    {
        transformedMeshes.push_back(meshes.at(i).clone());

        for(unsigned int vertex = 0; vertex < meshes.at(i).vertices.size(); vertex++)
        {
            boundingBoxMin.x = std::min(boundingBoxMin.x, meshes.at(i).vertices.at(vertex).x);
            boundingBoxMin.y = std::min(boundingBoxMin.y, meshes.at(i).vertices.at(vertex).y);
            boundingBoxMin.z = std::min(boundingBoxMin.z, meshes.at(i).vertices.at(vertex).z);

            boundingBoxMax.x = std::max(boundingBoxMax.x, meshes.at(i).vertices.at(vertex).x);
            boundingBoxMax.y = std::max(boundingBoxMax.y, meshes.at(i).vertices.at(vertex).y);
            boundingBoxMax.z = std::max(boundingBoxMax.z, meshes.at(i).vertices.at(vertex).z);
        }
    }

    float3 boundingBoxDimensions = boundingBoxMax - boundingBoxMin;
    float largestBoundingBoxSide = std::max(std::max(boundingBoxDimensions.x, boundingBoxDimensions.y), boundingBoxDimensions.z);

    renderMeshFractal(meshes, transformedMeshes, width, height, frameBuffer, depthBuffer, largestBoundingBoxSide, rank, size, depthLimit);//depthLimit);

    std::vector<unsigned char> partialFrameBuffer;
    partialFrameBuffer.resize(width * height * 4, 0);
    for (unsigned int i = 3; i < (4 * width * height); i += 4)
    {
        partialFrameBuffer.at(i) = 255;
    }

    std::vector<float> finalDepthBuffer;
    int count = width * height;
    finalDepthBuffer.resize(count, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    // use MPI_Allreduce
    MPI_Allreduce(&depthBuffer[0], &finalDepthBuffer[0], count, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

    for(int i = 0; i< partialFrameBuffer.size(); i=i+4){
      if(depthBuffer.at(i/4) == finalDepthBuffer.at(i/4)){
        partialFrameBuffer.at(i) = frameBuffer.at(i);
        partialFrameBuffer.at(i+1) = frameBuffer.at(i+1);
        partialFrameBuffer.at(i+2) = frameBuffer.at(i+2);
        partialFrameBuffer.at(i+3) = frameBuffer.at(i+3);
      }
    }
    count = width * height * 4;

    MPI_Reduce(&partialFrameBuffer[0], &frameBuffer[0], count, MPI_BYTE, MPI_BOR, 0,MPI_COMM_WORLD);

    return frameBuffer;
}
