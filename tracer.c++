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
#include "tracer.hh"

using namespace std;

int main() {
    // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
    SDL_Window *window = SDL_CreateWindow("SDL2/OpenGL Demo", 0, 0, WIDTH,
                                          HEIGHT, SDL_WINDOW_OPENGL);
    sdl_check();

    // Create an OpenGL context associated with the window.
    SDL_GLContext glcontext = SDL_GL_CreateContext(window);
    sdl_check();

    // Initalize OpenGL
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glewInit();
    gl_init_viewport(w, h);
    GLuint buffer_id = gl_init_buffer(w, h);
    GLuint texture_id = gl_create_texture(w, h);

    bool running = true;
    SDL_Event event;

    Sphere s0(float3(5, 5, 10), 2, float3(1, 0, 0));
    Sphere s1(float3(WIDTH / 2, HEIGHT / 2, 20), 1.5, float3(1, 0, 0));
    Sphere s2(float3(-5, -1, 15), 1, float3(1, 0, 0));
    Sphere s3(float3(5, 1, 30), 1, float3(1, 0, 0));
    // light source
    Sphere l0(float3(-10, 2, 10), 5, float3(1, 0, 0), 0, 0, float3(3));
    Sphere spheres[SPHERES] = {s0, s1, s2, s3, l0};

    // copy texture to CPU (not strictly necessary)
    size_t n_pixels = w * h * 4;
    float *pixels = new float[n_pixels];  // obv only do this once
    glGetTextureImage(texture_id, 0, GL_RGBA, GL_FLOAT,
                      n_pixels * sizeof(float), pixels);
    gl_check();

    for (int i = 0; i < n_pixels; ++i)
        pixels[i] = (rand() % 1000) / 1000.;
    // cpu_render(spheres, SPHERES, WIDTH, HEIGHT, pixels);

    // copy texture to GPU
    glBindTexture(GL_TEXTURE_2D, texture_id);
    gl_check();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); //unbind pixel unpack. IMPORTANT.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, pixels);
    gl_check();

    while (running) {
        auto t0 = chrono::high_resolution_clock::now();
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
        }

        // render buffer
        // gl_buf2tex(w, h, buffer_id, texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        gl_check();
        gl_draw_fullscreen();

        // glClearColor(0, 0, 0, 1);                            // set clear
        // color glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // clear
        // context
        SDL_GL_SwapWindow(window);  // update window

        // limit framerate
        auto t1 = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        // cout << dt << endl;
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
    glEnable(GL_TEXTURE_2D);  // enable texturing
    gl_check();
}

/**
 * @brief Initalizes the GL buffer.
 * @param[in] w desired width
 * @param[in] h desired height
 * @return the new buffer ID
 */
GLuint gl_init_buffer(int w, int h) {
    GLuint buffer_id;
    // get a buffer ID
    glGenBuffers(1, &buffer_id);
    gl_check();
    // set it as the current unpack buffer (a PBO)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id);
    gl_check();
    // allocate data for the buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4 * sizeof(float), nullptr,
                 GL_DYNAMIC_COPY);
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
    gl_check();
    // allocate texture memory
    glTextureStorage2D(texture_id, 1, GL_RGBA32F, w, h);
    gl_check();
    // set interpolation (must be nearest for RECTANGLE_ARB)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl_check();
    return texture_id;
}

/**
 * @brief Make an existing texture from an existing buffer
 * @details Used to display data rendered into a buffer
 * @param[in] w width of the texture/buffer
 * @param[in] h height of the texture/buffer
 * @param[in] buffer_id the buffer ID
 * @param[in] texture_id the texture ID
 */
void gl_buf2tex(int w, int h, GLuint buffer_id, GLuint texture_id) {
    // select buffer and texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id);
    gl_check();
    glBindTexture(GL_TEXTURE_2D, texture_id);
    gl_check();
    // make a texture from the buffer
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, nullptr);
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
