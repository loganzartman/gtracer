#ifndef CUDA_RENDER_CUH
#define CUDA_RENDER_CUH
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda_render.hh"
#define BLOCK_SIZE 256

cudaGraphicsResource_t cuda_buffer;
cudaGraphicsResource_t cuda_texture;
cudaStream_t cuda_stream;
void *device_geom_storage;
Geometry** device_geom;

struct CUDAKernelArgs {
    size_t w;
    size_t h;
    // Mat4f &camera;
    // std::vector<Geometry *> &geom;
    // AABB bounds;
    // const UniformGrid &grid;
    unsigned iteration;
    float *pixels;
};

template <typename II>
size_t cuda_geom_size(II b, II e);

template <typename II>
void cuda_geom_copy(II b, II e, void *dst, Geometry** items);

__global__ void cuda_render_kernel(CUDAKernelArgs args);

#endif