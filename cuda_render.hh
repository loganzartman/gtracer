#ifndef CUDA_RENDER_HH
#define CUDA_RENDER_HH
#include <GL/glew.h>
#include <cstddef>
#include <vector>
#include "Geometry.hh"
#include "Mat.hh"

void cuda_init(GLuint texture_id, GLuint buffer_id, GLuint display_buffer_id);

void cuda_render(size_t w, size_t h, const Mat4f& camera, Geometry* geom,
                 size_t geom_len, unsigned iteration, bool accel);
void cuda_destroy();

#endif