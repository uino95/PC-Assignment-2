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
    int angle,
    int depthLimit,
    float scale = 1.0,
    float3 distanceOffset = {0, 0, 0})
{
    // Start by rendering the mesh at this depth
    for (unsigned int j = 0; j < meshes.size(); j++)
    {
        Mesh &mesh = meshes.at(j);
        Mesh &transformedMesh = transformedMeshes.at(j);
        runVertexShader(mesh, transformedMesh, distanceOffset, scale, width, height);
        rasteriseTriangles(transformedMesh, frameBuffer, depthBuffer, width, height);
    }

    int i = 0;
    int currentDepth = 1;
    std::vector<float3> currentOffsets;
    currentOffsets.push_back(distanceOffset);
    std::vector<float3> tmpOffsets;
    updateList(currentOffsets, largestBoundingBoxSide, scale, tmpOffsets);
    currentOffsets = tmpOffsets;
    tmpOffsets.clear();
    scale = scale / 3.0;

    // Check whether we've reached the recursive depth of the fractal we want to reach
    while(currentDepth != depthLimit)
    {
        if(i < pow(3, 3 * currentDepth))
        {
            // We draw the new objects in a grid around the "main" one.
            // We thus skip the location of the object itself.
            if(currentOffsets.at(i) != 0)
            {
                for (unsigned int j = 0; j < meshes.size(); j++)
                {
                    Mesh &mesh = meshes.at(j);
                    Mesh &transformedMesh = transformedMeshes.at(j);
                    runVertexShader(mesh, transformedMesh, currentOffsets.at(i), scale, width, height);
                    rasteriseTriangles(transformedMesh, frameBuffer, depthBuffer, width, height);
                }
            }
        }
        // in order to avoid unuseful computation
        else if(currentDepth + 1 < depthLimit)
        {
            // Now we update the list of the offset in a smaller size
            updateList(currentOffsets, largestBoundingBoxSide, scale, tmpOffsets);
            currentOffsets = tmpOffsets;
            tmpOffsets.clear();
            currentDepth++;
            scale = scale / 3.0;
            i = -1;
        }
        else {
            return;
        }
        i++;
    }
}

// This function kicks off the rasterisation process.
std::vector<unsigned char> rasterise(std::vector<Mesh> &meshes, unsigned int width, unsigned int height, int angle, unsigned int depthLimit)
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

    std::cout << "Rendering image... " << std::flush;

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


    renderMeshFractal(meshes, transformedMeshes, width, height, frameBuffer, depthBuffer, largestBoundingBoxSide, angle, depthLimit);//depthLimit);


    std::cout << "finished!" << std::endl;

    return frameBuffer;


}