#include <GL/glew.h>
#include <iostream>
#include "render.cuh"

void cuda_render(GLuint buffer_id, size_t w, size_t h, const Mat4f& camera,
                 std::vector<Geometry*> geom, unsigned iteration) {
    using namespace std;
    cout << "Buffer ID: " << buffer_id << endl;
}