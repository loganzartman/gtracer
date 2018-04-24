#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <iostream>
#include "cuda_render.cuh"

void cuda_init(GLuint texture_id, GLuint buffer_id) {
    // register GL buffer and texture as CUDA resources
    cudaGraphicsGLRegisterBuffer(&cuda_buffer, buffer_id,
                                 cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterImage(&cuda_texture, texture_id, GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlagsNone);

    // create CUDA stream
}

void cuda_render(GLuint buffer_id, size_t w, size_t h, const Mat4f &camera,
                 std::vector<Geometry *> geom, unsigned iteration) {
    using namespace std;

    const size_t size_pixels = w * h;
    float *mem_ptr;
    cudaArray *array_ptr;

    // map resources
    cudaStreamCreate(&cuda_stream);
    cudaGraphicsMapResources(1, &cuda_buffer, cuda_stream);
    cudaGraphicsMapResources(1, &cuda_texture, cuda_stream);

    size_t size_mapped;
    cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_texture, 0, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&mem_ptr, &size_mapped,
                                         cuda_buffer);
    assert(size_mapped == size_pixels * 4 * sizeof(float));  // RGBA32F

    // run kernel
    const int num_blocks = (size_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_render_test_kernel<<<num_blocks, BLOCK_SIZE>>>(w, h, mem_ptr);

    // unmap resources
    cudaGraphicsUnmapResources(1, &cuda_buffer, cuda_stream);
    cudaGraphicsUnmapResources(1, &cuda_texture, cuda_stream);
    cudaStreamDestroy(cuda_stream);
}

void cuda_destroy() {}

__global__ void cuda_render_test_kernel(size_t w, size_t h, float *mem_ptr) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < w * h; i += stride) {
        const size_t idx = i * 4;
        const size_t x = i % w;
        const size_t y = i / w;
        mem_ptr[idx + 0] = (float)x / w;
        mem_ptr[idx + 1] = (float)y / w;
        mem_ptr[idx + 2] = 1.f - fabs(((float)y / h) - 0.5f) * 2;
        mem_ptr[idx + 3] = 1.f;
    }
}