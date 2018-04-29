#ifndef CUDA_RENDER_CUH
#define CUDA_RENDER_CUH
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "Geometry.hh"
#include "UniformGrid.hh"
#include "cuda_render.hh"
#define BLOCK_SIZE 256

cudaGraphicsResource_t cuda_buffer;
cudaGraphicsResource_t cuda_texture;
cudaStream_t cuda_stream;

struct CUDAKernelArgs {
    size_t w;
    size_t h;
    Mat4f camera;
    AABB bounds;
    UniformGrid grid;
    unsigned iteration;
    float *pixels;
};

__global__ void cuda_render_kernel(CUDAKernelArgs args);
__device__ Float3 cuda_trace(const Float3 &ray_orig, const Float3 &ray_dir,
                             AABB world_bounds, const UniformGrid &grid,
                             int depth);
__device__ bool cuda_ray_intersect(const Float3 &ray_orig,
                                   const Float3 &ray_dir, AABB world_bounds,
                                   const UniformGrid &grid,
                                   Float3 &intersection, Geometry *&hit_geom);
template <typename II>
__device__ bool cuda_ray_intersect_items(const Float3 &ray_orig,
                                         const Float3 &ray_dir, II b, II e,
                                         Float3 &intersection,
                                         Geometry *&hit_geom);
#endif
