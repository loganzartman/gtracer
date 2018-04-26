#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <iostream>
#include "cuda_render.cuh"

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

    // copy geometry data to device
    const size_t geom_storage_size = cuda_geom_size(geom.begin(), geom.end());
    cudaMalloc(&device_geom_storage, geom_storage_size); // structs themselves
    cudaMalloc(&device_geom, geom.size() * sizeof(Geometry*)); // list of Geometry*
    cuda_geom_copy(geom.begin(), geom.end(), device_geom_storage, device_geom);
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
    assert(size_mapped == size_pixels * 4 * sizeof(float));  // RGBA32F

    // run kernel
    CUDAKernelArgs args = {w, h, iteration, mem_ptr};
    const int num_blocks = (size_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_render_kernel<<<num_blocks, BLOCK_SIZE>>>(args);
}

/**
 * @brief Compute the size occupied by many Geometry structs.
 * @detail This is the total *actual size* of underlying Geometries, not the
 * pointers to them.
 * 
 * @tparam II     an Input Iterator
 * @param b       the beginning iterator of geometry
 * @param e       the ending iterator of geometry
 * @return size_t size in bytes
 */
template <typename II>
size_t cuda_geom_size(II b, II e) {
    size_t total_size = 0;
    while (b != e) {
        const size_t item_size = sizeof(decltype(**b));
        total_size += item_size;
        ++b;
    }
    return total_size;
}

/**
 * @brief Copies geometries to GPU and populates an array with pointers to them
 * @detail The GPU must store both an array of actual Geometry structs of 
 * varying type and an array of pointers to each one, as it would be impossible
 * to traverse the array of differently-typed objects.
 * 
 * @tparam II   an Input Iterator 
 * @param b     the beginning iterator of geometry
 * @param e     the ending iterator of geometry
 * @param dst   start of the array storing actual Geometry structs
 * @param items start of the array storing Geometry pointers
 */
template <typename II>
void cuda_geom_copy(II b, II e, void *dst, Geometry** items) {
    uint8_t* dst8 = (uint8_t*)dst;
    while (b != e) {
        const size_t item_size = sizeof(decltype(**b));
        cudaMemcpy(dst, *b, item_size, cudaMemcpyHostToDevice);
        cudaMemcpy(items, (Geometry*)dst, sizeof(Geometry*), cudaMemcpyHostToDevice);

        dst8 += item_size; 
        ++items;
        ++b;
    }
}

/**
 * @brief Destroy resource
 * 
 */
void cuda_destroy() {
    // free memory
    cudaFree(device_geom_storage);
    cudaFree(device_geom);

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