#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "Mat.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "Vec3.hh"
#include "render.hh"
#include "tracer.hh"
#include "transform.hh"

using namespace std;

int main() {
    // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
    SDL_Window *window = SDL_CreateWindow("SDL2/OpenGL Demo", 0, 0, WIDTH,
                                          HEIGHT, SDL_WINDOW_OPENGL);
    sdl_check();

    // Create an OpenGL context associated with the window.
    SDL_GLContext glcontext = SDL_GL_CreateContext(window);
    sdl_check();
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // Initalize OpenGL
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glewInit();
    gl_init_viewport(w, h);
    GLuint buffer_id = gl_create_buffer(w, h);  // for gpu
    GLuint texture_id = gl_create_texture(w, h);

    bool running = true;
    SDL_Event event;

    // state for orbit controls
    float mouse_x = 0, mouse_y = 0;
    float3 orbit_pos(0.3, 0, 0);
    float3 orbit_vel;
    float orbit_zoom = 30;

    // scene geometry
    vector<Sphere> spheres = construct_spheres(SPHERES);

    // prepare CPU pixel buffer
    size_t n_pixels = w * h * 4;
    float *pixels = new float[n_pixels];  // obv only do this once
    fill_n(pixels, n_pixels, 0);

    while (running) {
        auto t0 = chrono::high_resolution_clock::now();

        // poll SDL events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            else if (event.type == SDL_MOUSEMOTION) {
                float x = event.motion.x;
                float y = event.motion.y;
                if (event.motion.state & SDL_BUTTON_LMASK)
                    orbit_vel +=
                        (float3(y, x, 0) - float3(mouse_y, mouse_x, 0)) * 0.005;
                mouse_x = x;
                mouse_y = y;
            } else if (event.type == SDL_MOUSEWHEEL) {
                orbit_zoom -= event.wheel.y * 2;
            }
        }

        // integrate orbit controls velocity
        orbit_pos += orbit_vel;
        orbit_pos.x = max(-(float)M_PI / 2, min((float)M_PI / 2, orbit_pos.x));
        orbit_vel *= 0.5;

        // compute orbit camera transform
        Mat4f camera = Mat4f::identity();
        camera = camera * transform_rotateY(orbit_pos.y);
        camera = camera * transform_rotateX(-orbit_pos.x);
        camera = camera * transform_translate(float3(0, 0, orbit_zoom));

        // do raytracing
        cpu_render(pixels, w, h, camera, spheres);

        // copy texture to GPU
        // gl_buf2tex(w, h, buffer_id, texture_id); // only necessary for gpu
        gl_data2tex(w, h, pixels, texture_id);

        // render buffer
        gl_draw_tex(texture_id);

        SDL_GL_SwapWindow(window);  // update window

        // limit framerate
        auto t1 = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        cout << "\e[1G\e[0K" << (1000 / dt) << "fps" << flush;
        SDL_Delay(max(0l, 1000 / TARGET_FPS - dt));
    }

    // Once finished with OpenGL functions, the SDL_GLContext can be deleted.
    SDL_GL_DeleteContext(glcontext);
    sdl_check();

    delete[] pixels;

    return 0;
}

/**
 * @brief Initalizes the viewport
 * @param[in] w desired width
 * @param[in] h desired height
 */
void gl_init_viewport(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    glOrtho(0.f, 1.f, 0.f, 1.f, -1.f, 1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_DEPTH_TEST);  // do we really need depth sorting?
    gl_check();
}

/**
 * @brief Creates a pixel unpack buffer.
 * @param[in] w desired width
 * @param[in] h desired height
 * @return the new buffer ID
 */
GLuint gl_create_buffer(int w, int h) {
    GLuint buffer_id;
    // get a buffer ID
    glGenBuffers(1, &buffer_id);
    // set it as the current unpack buffer (a PBO)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id);
    // allocate data for the buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4 * sizeof(float), nullptr,
                 GL_DYNAMIC_COPY);
    // unbind
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    gl_check();
    return buffer_id;
}

/**
 * @brief Creates a GL texture of specified size.
 * @param[in] w desired width
 * @param[in] h desired height
 * @return the new texture ID
 */
GLuint gl_create_texture(int w, int h) {
    GLuint texture_id;
    // get a texture ID
    glCreateTextures(GL_TEXTURE_2D, 1, &texture_id);
    // allocate texture memory
    glTextureStorage2D(texture_id, 1, GL_RGBA32F, w, h);
    // set interpolation (must be nearest for RECTANGLE_ARB)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl_check();
    return texture_id;
}

/**
 * @brief Copy an existing buffer to an existing texture
 * @details Used to display data rendered into a buffer
 * @param[in] buffer_id the buffer ID
 * @param[in] texture_id the texture ID
 * @param[in] w width of the texture/buffer
 * @param[in] h height of the texture/buffer
 */
void gl_buf2tex(GLuint buffer_id, GLuint texture_id, int w, int h) {
    // select buffer and texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    // copy buffer to texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, nullptr);
    // unbind
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    gl_check();
}

/**
 * @brief Copy pixel data on the CPU to an existing texture
 * @details Used to display data rendered into a buffer.
 * The pixel data must be in RGBA float32 format.
 * @param[in] buffer_id the buffer ID
 * @param[in] texture_id the texture ID
 * @param[in] w width of the texture/buffer
 * @param[in] h height of the texture/buffer
 */
void gl_data2tex(int w, int h, float *pixels, GLuint texture_id) {
    // bind texture, unbind pixel unpack buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);  // muy importante!
    glBindTexture(GL_TEXTURE_2D, texture_id);
    // copy pixels to texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, pixels);
    // unbind
    glBindTexture(GL_TEXTURE_2D, 0);
    gl_check();
}

void gl_draw_tex(GLuint texture_id) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // clear screen
    glBindTexture(GL_TEXTURE_2D, texture_id);            // bind texture
    glEnable(GL_TEXTURE_2D);                             // enable texturing
    gl_draw_fullscreen();                                // draw fullscreen quad
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    gl_check();
}

/**
 * @brief Draws a fullscreen quad.
 */
void gl_draw_fullscreen() {
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex3f(0, 0, 0);

    glTexCoord2f(0, 0);
    glVertex3f(0, 1, 0);

    glTexCoord2f(1, 0);
    glVertex3f(1, 1, 0);

    glTexCoord2f(1, 1);
    glVertex3f(1, 0, 0);
    glEnd();
    gl_check();
}

// 3D Object Creation

/**
 * @brief Construct an array of spheres to render
 * @param[in] num_spheres used to specify the amount
 * of spheres in the constructed array
 * @return A pointer to the constructed array
 */
vector<Sphere> construct_spheres(size_t num_spheres) {
    //surface color, transparency, reflectivity, emission color
    Material wood(float3(0.545, 0.271, 0.075), 0, 0.2, float3(0));
    Material metal(float3(0.377, 0.377, 0.377), 0, 0.8, float3(0));
    Material mirror(float3(0.8, 0.8, 1.0), 0, 1.0, float3(0));

    vector<Sphere> spheres;
    // position, radius, material    
    spheres.push_back(Sphere(float3(0.0, -10004, -20), 10000, &metal));
                             
    spheres.push_back(
        Sphere(float3(0.0, 0, 0), 4, &wood));
    spheres.push_back(
        Sphere(float3(5.0, -1, 5), 2, &mirror));
    spheres.push_back(
        Sphere(float3(5.0, 0, -5), 3, &metal));
    spheres.push_back(
        Sphere(float3(-5.5, 0, 5), 3, &wood));
    // light
    spheres.push_back(Sphere(float3(0.0, 20, 5), 3, &mirror));

    return spheres;
}

// Util

void output_to_ppm(float3 *image, size_t w, size_t h) {
    std::ofstream ofs("./output.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (unsigned i = 0; i < w * h; ++i) {
        ofs << (unsigned char)(std::min(float(1), image[i].x) * 255)
            << (unsigned char)(std::min(float(1), image[i].y) * 255)
            << (unsigned char)(std::min(float(1), image[i].z) * 255);
    }
    ofs.close();
}
