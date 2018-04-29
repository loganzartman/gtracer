#ifndef CUDA_RENDER_CUH
#define CUDA_RENDER_CUH
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda_render.hh"
#include "Geometry.hh"
#include "UniformGrid.hh"
#define BLOCK_SIZE 256

cudaGraphicsResource_t cuda_buffer;
cudaGraphicsResource_t cuda_texture;
cudaStream_t cuda_stream;

struct CUDAKernelArgs {
    size_t w;
    size_t h;
    const Mat4f &camera;
    AABB bounds;
    const UniformGrid &grid;
    unsigned iteration;
    float *pixels;
    Geometry **geom;
    size_t geom_len;
};

void cuda_update_geometry(const std::vector<Geometry*>& geom, Geometry** dev_geom);

__global__ void cuda_render_kernel(CUDAKernelArgs args);
__device__ Float3 cuda_trace(Float3 ray_orig, Float3 ray_dir,
                 Geometry **geom, AABB world_bounds,
                 const UniformGrid &grid, int depth);
#endif
