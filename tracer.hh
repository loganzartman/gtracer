#ifndef TRACER_HH
#define TRACER_HH

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "Sphere.hh"
#include "Vec3.hh"
#include "render.hh"

#define TARGET_FPS 60
#define SPHERES 5
#define WIDTH 640
#define HEIGHT 480
#define ERRINFO __FILE__ << ":" << __func__ << ":" << __LINE__

void gl_init_viewport(int w, int h);
GLuint gl_init_buffer(int w, int h);
GLuint gl_create_texture(int w, int h);
void gl_buf2tex(int w, int h, GLuint buffer_id, GLuint texture_id);
void gl_draw_fullscreen();
#define gl_check()                                                  \
    ({                                                              \
        GLenum err;                                                 \
        while ((err = glGetError()))                                \
            cout << ERRINFO << ": gl err 0x" << hex << err << endl; \
    })

void output_to_ppm(float3 *image, size_t w, size_t h);
#define sdl_check()                       \
    ({                                    \
        const char *err = SDL_GetError(); \
        if (err[0] != 0) {                \
            cout << ERRINFO << ": ";      \
            cout << err << endl;          \
            SDL_ClearError();             \
        }                                 \
    })

#endif