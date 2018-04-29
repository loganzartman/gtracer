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
#include <string>
#include <unordered_map>
#include <vector>

#include "Box.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "Tri.hh"
#include "Vec3.hh"
#include "render.hh"

#define color_float_to_byte(f) \
    (static_cast<uint8_t>(std::min(255.f, std::max(0.f, f * 255))))

#define TARGET_FPS 60
#define WIDTH 640
#define HEIGHT 480
#define ERRINFO __FILE__ << ":" << __func__ << ":" << __LINE__

void gl_init_viewport(int w, int h);
GLuint gl_create_buffer(int w, int h);
GLuint gl_create_texture(int w, int h);
void gl_buf2tex(int w, int h, GLuint buffer_id, GLuint texture_id);
void gl_data2tex(int w, int h, float *pixels, GLuint texture_id);
void gl_draw_tex(GLuint texture_id);
void gl_draw_fullscreen();
#define gl_check()                                                  \
    ({                                                              \
        GLenum err;                                                 \
        while ((err = glGetError()))                                \
            cout << ERRINFO << ": gl err 0x" << hex << err << endl; \
    })

void output_bmp(float *pixels, int w, int h, std::string outfile);
void output_time(double total_time, unsigned iteration, size_t geom_size,
                 unsigned long rays_cast, bool gpu, bool accel);

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
