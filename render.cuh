#ifndef RENDER_CUH
#define RENDER_CUH
#include <GL/glew.h>
#include <cstddef>
#include <vector>
#include "Geometry.hh"
#include "Mat.hh"

void cuda_render(GLuint buffer_id, size_t w, size_t h, const Mat4f& camera,
                 std::vector<Geometry*> geom, unsigned iteration);

#endif