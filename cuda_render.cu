#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <iostream>
#include <vector>
#include "cuda_render.cuh"
#include "Geometry.hh"
#include "Sphere.hh"
#include "Tri.hh"
#include "Box.hh"

/**
 * @brief Initializes CUDA resources.
 * @detail Called once upon program start. Registers GL texture and buffer
 * for CUDA/GL interop; creates stream and maps buffer and texture to stream.
 * 
 * @param texture_id ID of the GL texture
 * @param buffer_id  ID of the GL buffer
 */
void cuda_init(GLuint texture_id, GLuint buffer_id) {
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
                 Geometry** geom, size_t geom_len, unsigned iteration) {
    using namespace std;

    const size_t size_pixels = w * h;
    float *mem_ptr;
    cudaArray *array_ptr;

    size_t size_mapped;
    cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_texture, 0, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&mem_ptr, &size_mapped,
                                         cuda_buffer);
    //assert(size_mapped == size_pixels * 4 * sizeof(float));  // RGBA32F

    // run kernel
    CUDAKernelArgs args = {w, h, iteration, mem_ptr, geom, geom_len};
    const int num_blocks = (size_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_render_kernel<<<num_blocks, BLOCK_SIZE>>>(args);
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

    const size_t len = args.w * args.h;
    for (size_t i = index; i < len; i += stride) {
        const size_t idx = i * 4;
        const size_t x = i % args.w;
        const size_t y = i / args.w;
        float fx = (float)x / args.w, fy = (float)y / args.h;
        args.pixels[idx + 0] = sin(fx + args.iteration * 0.1f);
        args.pixels[idx + 1] = cos(fy + args.iteration * 0.02f);
        args.pixels[idx + 2] = 0.f;
        args.pixels[idx + 3] = 1.f;
    }
}
