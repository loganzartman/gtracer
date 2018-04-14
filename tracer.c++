#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Mat.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "Vec3.hh"
#include "options.hh"
#include "render.hh"
#include "tracer.hh"
#include "transform.hh"
#include "util.hh"

using namespace std;

int main(int argc, char *argv[]) {
    TracerArgs args = parse_args(argc, argv);

    // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
    SDL_Window *window =
        SDL_CreateWindow("raytracer", 0, 0, args.width, args.height, SDL_WINDOW_OPENGL);
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
    float orbit_zoom = 30;
    float3 trans(0);
    

    // scene geometry
    // surface color, transparency, reflectivity, emission color
    unordered_map<string, Material *> mats = {
        {"ground", new Material(float3(0.8, 0.8, 0.9), 0, 0.0, float3(0))},
        {"wood", new Material(float3(0.545, 0.271, 0.075), 0, 0.0, float3(0))},
        {"metal", new Material(float3(0.377, 0.377, 0.377), 0, 0.0, float3(0))},
        {"mirror", new Material(float3(0.8, 0.8, 1.0), 0, 1.0, float3(0))},
        {"light", new Material(float3(1), 0, 0, float3(1))}};
    vector<Sphere> spheres = construct_spheres_random(mats);

    // prepare CPU pixel buffer
    size_t n_pixels = w * h * 4;
    float *pixels = new float[n_pixels];  // obv only do this once
    fill_n(pixels, n_pixels, 0);

    unsigned iteration = 0;
    while (running) {
        auto t0 = chrono::high_resolution_clock::now();
    Mat4f camera = Mat4f::identity();

        // poll SDL events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            else if (event.type == SDL_MOUSEMOTION) {
                float x = event.motion.x;
                float y = event.motion.y;
                if (event.motion.state & SDL_BUTTON_LMASK) {
                    iteration = 0;  // reset progress
                    orbit_pos +=
                        (float3(y, x, 0) - float3(mouse_y, mouse_x, 0)) * 0.005;
                }
                mouse_x = x;
                mouse_y = y;
            } else if (event.type == SDL_MOUSEWHEEL) {
                iteration = 0;  // reset progress
                orbit_zoom -= event.wheel.y * 2;
            } else if (event.type == SDL_KEYDOWN) {
                bool reset = true;
                switch (event.key.keysym.sym) {
                    case SDLK_LEFT:
                        trans.x += 1;
                        break;
                    case SDLK_RIGHT:
                        trans.x -= 1;
                        break;
                    case SDLK_UP:
                        trans.z += 1;
                        break;
                    case SDLK_DOWN:
                        trans.z -= 1;
                        break;
                    default:
                        reset = false;
                        break;
                }
                if (reset)
                    iteration = 0;  // reset progress
            }
        }

        // limit orbit controls
        orbit_pos.x = max(-(float)M_PI / 2, min((float)M_PI / 2, orbit_pos.x));

        // compute orbit camera transform
        camera = camera * transform_translate(float3(trans.x, 0, trans.z));
        camera = camera * transform_rotateY(orbit_pos.y);
        camera = camera * transform_rotateX(-orbit_pos.x);
        camera = camera * transform_translate(float3(0, 0, orbit_zoom));

        // do raytracing
        cpu_render(pixels, w, h, camera, spheres, iteration);

        // copy texture to GPU
        // gl_buf2tex(w, h, buffer_id, texture_id); // only necessary for gpu
        gl_data2tex(w, h, pixels, texture_id);

        // render buffer
        gl_draw_tex(texture_id);

        SDL_GL_SwapWindow(window);  // update window

        // check for completion
        ++iteration;
        if (args.iterations > 0 && iteration >= args.iterations)
            break;

        // limit framerate
        auto t1 = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        cout << "\e[1G\e[0K"
             << "iteration " << iteration << ". " << (1000 / dt) << "fps"
             << flush;
        SDL_Delay(max(0l, 1000 / TARGET_FPS - dt));
    }

    // output rendered image
    if (args.output) {
        // TODO: copy data back from GPU when in GPU mode
        output_bmp(pixels, w, h, args.outfile);
    }

    // Once finished with OpenGL functions, the SDL_GLContext can be deleted.
    SDL_GL_DeleteContext(glcontext);
    sdl_check();

    // deallocate materials
    for (auto it = mats.begin(); it != mats.end(); ++it)
        delete it->second;

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
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    // allocate texture memory
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
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
vector<Sphere> construct_spheres(unordered_map<string, Material *> mats) {
    vector<Sphere> spheres;
    // position, radius, material
    spheres.push_back(Sphere(float3(0.0, -10004, -20), 10000, mats["metal"]));

    spheres.push_back(Sphere(float3(0.0, 0, 0), 4, mats["mirror"]));
    spheres.push_back(Sphere(float3(5.0, -1, 5), 2, mats["mirror"]));
    spheres.push_back(Sphere(float3(5.0, 0, -5), 3, mats["metal"]));
    spheres.push_back(Sphere(float3(-5.5, 0, 5), 3, mats["wood"]));
    // light
    spheres.push_back(Sphere(float3(0.0, 20, 5), 3, mats["light"]));

    return spheres;
}

vector<Sphere> construct_spheres_random(
    unordered_map<string, Material *> mats) {
    vector<Sphere> spheres;

    // position, radius, material
    spheres.push_back(Sphere(float3(0.0, -10000, -20), 10000, mats["ground"]));

    for (int i = 0; i < 20; ++i) {
        float3 pos(randf(-20, 20), randf(0, 10), randf(-20, 20));
        float radius = randf(2, 5);
        float3 col(randf(0, 1), randf(0, 1), randf(0, 1));
        Material *mat = new Material(col, randf(0,1), randf(0, 0.5), float3(0));
        mats[to_string(i)] = mat;
        spheres.push_back(Sphere(pos, radius, mat));
    }

    return spheres;
}

/**
 * @brief Output a pixel array to a bitmap
 *
 * @param[in] pixels an RGBA float32 array
 * @param[in] w width of the input array in pixels
 * @param[in] h height of the input array in pixels
 * @param[in] outfile path to output to
 */
void output_bmp(float *pixels, int w, int h, string outfile) {
    SDL_Surface *surf =
        SDL_CreateRGBSurface(0, w, h, 24, 0, 0, 0, 0);
    sdl_check();

    // copy pixels to surface
    uint8_t *spixels = static_cast<uint8_t *>(surf->pixels);
    for (int i = 0; i < w * h; ++i) {
        spixels[i * 3 + 0] = color_float_to_byte(pixels[i * 4 + 0]);
        spixels[i * 3 + 1] = color_float_to_byte(pixels[i * 4 + 1]);
        spixels[i * 3 + 2] = color_float_to_byte(pixels[i * 4 + 2]);
    }

    // write bitmap
    SDL_SaveBMP(surf, outfile.c_str());
    sdl_check();

    SDL_FreeSurface(surf);
    sdl_check();
}
