#ifndef CUDA_RENDER_CUH
#define CUDA_RENDER_CUH
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda_render.hh"

cudaGraphicsResource_t cuda_buffer;
cudaGraphicsResource_t cuda_texture;
cudaStream_t cuda_stream;

__global__
void cuda_render_test_kernel();

#endif