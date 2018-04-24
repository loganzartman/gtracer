#include <GL/glew.h>
#include <iostream>
#include <cuda_gl_interop.h>
#include "cuda_render.cuh"

void cuda_init(GLuint texture_id, GLuint buffer_id) {
    // register GL buffer and texture as CUDA resources
    cudaGraphicsGLRegisterBuffer(&cuda_buffer, buffer_id, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterImage(&cuda_texture, texture_id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    // create CUDA stream
    cudaStreamCreate(&cuda_stream);
}

void cuda_render(GLuint buffer_id, size_t w, size_t h, const Mat4f& camera,
    std::vector<Geometry*> geom, unsigned iteration) {
    using namespace std;   
    
    size_t size;
    unsigned char *mem_ptr;
    cudaArray *array_ptr;

    // map resources 
    cudaGraphicsMapResources(1, &cuda_buffer, cuda_stream);
    cudaGraphicsMapResources(1, &cuda_texture, cuda_stream);
    
    cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_texture, 0, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&mem_ptr, &size, cuda_buffer);
    cuda_render_test_kernel<<<1, 1>>>();
    
    // unmap resources
    cudaGraphicsUnmapResources(1, &cuda_buffer, cuda_stream);
    cudaGraphicsUnmapResources(1, &cuda_texture, cuda_stream);
}

void cuda_destroy() {
    cudaStreamDestroy(cuda_stream);
}

__global__
void cuda_render_test_kernel() {

}