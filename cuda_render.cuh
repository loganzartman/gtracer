#ifndef CUDA_RENDER_CUH
#define CUDA_RENDER_CUH
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda_render.hh"
#include "Geometry.hh"
#define BLOCK_SIZE 256

cudaGraphicsResource_t cuda_buffer;
cudaGraphicsResource_t cuda_texture;
cudaStream_t cuda_stream;

struct CUDAKernelArgs {
    size_t w;
    size_t h;
    // Mat4f &camera;
    // AABB bounds;
    // const UniformGrid &grid;
    unsigned iteration;
    float *pixels;
    Geometry **geom;
    size_t geom_len;
};

void cuda_update_geometry(const std::vector<Geometry*>& geom, Geometry** dev_geom);

__global__ void cuda_render_kernel(CUDAKernelArgs args);
#endif
