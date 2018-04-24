#ifndef CUDA_RENDER_CUH
#define CUDA_RENDER_CUH
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda_render.hh"
#define BLOCK_SIZE 256

cudaGraphicsResource_t cuda_buffer;
cudaGraphicsResource_t cuda_texture;
cudaStream_t cuda_stream;

__global__ void cuda_render_test_kernel(size_t w, size_t h, float *mem_ptr);

#endif