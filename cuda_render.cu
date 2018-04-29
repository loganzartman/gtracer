#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "cuda_render.cuh"
#include "Geometry.hh"
#include "UniformGrid.hh"
#include "Mat.hh"
#include "Material.hh"
#include "Vec3.hh"
#include "transform.hh"
#include "util.hh"
#include "Sphere.hh"
#include "Tri.hh"
#include "Box.hh"

#define PRIMARY_RAYS 1

/**
 * @brief Initializes CUDA resources.
 * @detail Called once upon program start. Registers GL texture and buffer
 * for CUDA/GL interop; creates stream and maps buffer and texture to stream.
 * 
 * @param texture_id ID of the GL texture
 * @param buffer_id  ID of the GL buffer
 */
void cuda_init(GLuint texture_id, GLuint buffer_id, const std::vector<Geometry*>& geom) {
    // register GL buffer and texture as CUDA resources
    cudaGraphicsGLRegisterBuffer(&cuda_buffer, buffer_id,
                                 cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterImage(&cuda_texture, texture_id, GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlagsNone);

    // create CUDA stream
    cudaStreamCreate(&cuda_stream);

    // map resources
    cudaGraphicsMapResources(1, &cuda_buffer, cuda_stream);
    cudaGraphicsMapResources(1, &cuda_texture, cuda_stream);
}

void cuda_render(GLuint buffer_id, size_t w, size_t h, const Mat4f &camera,
                 std::vector<Geometry *> geom, unsigned iteration) {
    using namespace std;

    const size_t size_pixels = w * h;
    float *mem_ptr;
    cudaArray *array_ptr;

    size_t size_mapped;
    cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_texture, 0, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&mem_ptr, &size_mapped,
                                         cuda_buffer);
    //assert(size_mapped == size_pixels * 4 * sizeof(float));  // RGBA32F

    Geometry **dev_geom;
    cudaMallocManaged(&dev_geom, geom.size());

    cuda_update_geometry(geom, dev_geom);

    // run kernel
    CUDAKernelArgs args = {w, h, iteration, mem_ptr, dev_geom};
    const int num_blocks = (size_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_render_kernel<<<num_blocks, BLOCK_SIZE>>>(args);
}

/**
 * @brief Copies a new set of Geometry to the GPU
 * @detail allocates memory and uses kernel to construct Geometries on GPU
 * 
 * @param geom the vector of geometry to copy
 */
void cuda_update_geometry(const std::vector<Geometry*>& geom, Geometry** dev_geom) {
    Geometry *x;
    for (size_t i = 0; i < geom.size(); ++i) {
        cudaMallocManaged(&x, sizeof(decltype(*geom[i])));
        dev_geom[i] = x;
    }
}

/**
 * @brief Destroy resource
 * 
 */
void cuda_destroy() {
    // unmap resources
    cudaGraphicsUnmapResources(1, &cuda_buffer, cuda_stream);
    cudaGraphicsUnmapResources(1, &cuda_texture, cuda_stream);
    cudaStreamDestroy(cuda_stream);
}

/**
 * @brief Path tracing kernel
 * @param args current state 
 */
__global__ void cuda_render_kernel(CUDAKernelArgs args) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float inv_w = 1 / float(args.w);
    float inv_h = 1 / float(args.h);
    float fov = 30;
    float aspect_ratio = float(args.w) / float(args.h);
    float angle = tan(0.5 * M_PI * fov / 180.0);

    Mat4f dir_camera = transform_clear_translate(args.camera);
    Float3 origin = args.camera * Float3();

    const size_t len = args.w * args.h;
    for (size_t i = index; i < len; i += stride) {
        const size_t idx = i * 4;
        const size_t x = i % args.w;
        const size_t y = i / args.w;

        Float3 color;
        for (size_t i = 0; i < PRIMARY_RAYS; ++i) {
            float v_x =
                (2 * ((x + util::randf(0, 1)) * inv_w) - 1) * angle * aspect_ratio;
            float v_y = (1 - 2 * ((y + util::randf(0, 1)) * inv_h)) * angle;
            Float3 ray_dir = dir_camera * Float3(v_x, v_y, -1);
            ray_dir.normalize();

            color += cuda_trace(origin, ray_dir, args.geom, args.bounds,
                               args.grid, 8);
        }
        color *= 1.f / PRIMARY_RAYS;

        // compute all-time average color
        Float3 dst = Float3(args.pixels[idx], args.pixels[idx + 1],
                            args.pixels[idx + 2]);
        float f = 1;
        if (args.iteration > 0)
            f = 1.f / args.iteration;
        Float3 blended = color * f + dst * (1 - f);

        // write color
        args.pixels[idx] = blended.x;
        args.pixels[idx + 1] = blended.y;
        args.pixels[idx + 2] = blended.z;
        args.pixels[idx + 3] = 1;  // alpha
    }
}

__device__ Float3 cuda_trace(float *ray_orig, float *ray_dir,
                 Geometry **geom, AABB world_bounds,
                 const UniformGrid &grid, int depth) {
    return Float3(1);
}
