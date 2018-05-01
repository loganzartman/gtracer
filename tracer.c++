#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Box.hh"
#include "Geometry.hh"
#include "Mat.hh"
#include "Material.hh"
#include "Sphere.hh"
#include "Tri.hh"
#include "Vec3.hh"
#include "cuda_render.hh"
#include "loader.hh"
#include "options.hh"
#include "render.hh"
#include "tracer.hh"
#include "transform.hh"
#include "util.hh"
#include "raytracing.hh"

using namespace std;

int main(int argc, char *argv[]) {
    TracerArgs args = parse_args(argc, argv);

    // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
    SDL_Window *window = SDL_CreateWindow("raytracer", 0, 0, args.width,
                                          args.height, SDL_WINDOW_OPENGL);
    sdl_check();

    // Create an OpenGL context associated with the window.
    SDL_GLContext glcontext = SDL_GL_CreateContext(window);
    sdl_check();
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    bool running = true;
    SDL_Event event;

    // state for orbit controls
    float mouse_x = 0, mouse_y = 0;
    Float3 orbit_pos(0.3, 0, 0);
    float orbit_zoom = 30;
    Float3 trans(0);

    // surface color, transparency, reflectivity, emission color
    vector<Material> mats{
        Material(Float3(0.990, 0.990, 0.990), 0.0, 0.0, Float3(0)),  // white
        Material(Float3(0.950, 0.123, 0.098), 0.0, 0.0, Float3(0)),  // red
        Material(Float3(1, 0, 0), 0, 0, Float3(30, 30, 30)),           // lightr
        Material(Float3(1, 0, 0), 0, 0, Float3(0, 30, 0)),           // lightg
        Material(Float3(1, 0, 0), 0, 0, Float3(0, 0, 30))            // lightb
    };
    Material *mat_array = (Material *)util::hostdev_alloc(
        mats.size() * sizeof(Material), args.gpu);
    copy(mats.begin(), mats.end(), mat_array);

    // load geometry
    unsigned long last_modified = 0;
    vector<Geometry> geom;
    load(args.infile, geom, 100, &mat_array[1]);
    geom.push_back(
        Geometry(BoxData{AABB(Float3(-12, 2, -12), Float3(12, 1.8, 12))},
                 &mat_array[0]));
    geom.push_back(
        Geometry(SphereData{Float3(-20, 20, -20), 7}, &mat_array[2]));
    // geom.push_back(Geometry(SphereData{Float3(0, 20, 20), 7},
    // &mat_array[3])); geom.push_back(Geometry(SphereData{Float3(20, 20, 20),
    // 7}, &mat_array[4]));

    Geometry *geom_array = (Geometry *)util::hostdev_alloc(
        geom.size() * sizeof(Geometry), args.gpu);
    copy(geom.begin(), geom.end(), geom_array);

    // Initalize OpenGL
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glewInit();
    gl_init_viewport(w, h);

    // initialize GL buffers, textures, Cuda interop
    GLuint buffer_id = 0, display_buffer_id = 0, texture_id = 0;
    texture_id = gl_create_texture(w, h);
    if (args.gpu) {
        buffer_id = gl_create_buffer(w, h);
        display_buffer_id = gl_create_buffer(w, h);
        cuda_init(texture_id, buffer_id, display_buffer_id);
    }

    // initialize CPU pixel buffers
    float *pixels = nullptr;
    float *display_pixels = nullptr;
    if (!args.gpu) {
        size_t n_pixels = w * h * 4;
        pixels = new float[n_pixels];
        fill_n(pixels, n_pixels, 0);
        display_pixels = new float[n_pixels];
        fill_n(display_pixels, n_pixels, 0);
    }

    unsigned iteration = 0;
    auto start_time = chrono::high_resolution_clock::now();

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
                    start_time = chrono::high_resolution_clock::now();
                    orbit_pos +=
                        (Float3(y, x, 0) - Float3(mouse_y, mouse_x, 0)) * 0.005;
                }
                mouse_x = x;
                mouse_y = y;
            } else if (event.type == SDL_MOUSEWHEEL) {
                iteration = 0;  // reset progress
                start_time = chrono::high_resolution_clock::now();
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
                start_time = chrono::high_resolution_clock::now();
            }
        }

        // limit orbit controls
        orbit_pos.x = util::max(-(float)M_PI / 2,
                                util::min((float)M_PI / 2, orbit_pos.x));

        // compute orbit camera transform
        camera = camera * transform_translate(Float3(trans.x, 0, trans.z));
        camera = camera * transform_rotateY(orbit_pos.y);
        camera = camera * transform_rotateX(-orbit_pos.x);
        camera = camera * transform_translate(Float3(0, 0, orbit_zoom));

        // do raytracing
        if (args.gpu) {
            // GPU rendering mode
            cuda_render(w, h, camera, geom_array, geom.size(),
                        iteration, args.accel);
            gl_buf2tex(w, h, display_buffer_id, texture_id);  // copy buffer to texture
        } else {
            // CPU rendering mode
            cpu_render(pixels, display_pixels, w, h, camera, geom_array,
                       geom_array + geom.size(), iteration, args.threads,
                       args.accel);
            gl_data2tex(w, h, display_pixels, texture_id);  // copy buffer to texture
        }

        gl_draw_tex(texture_id);    // render buffer
        SDL_GL_SwapWindow(window);  // update window

        // check for completion
        ++iteration;
        if (args.iterations > 0 && iteration >= args.iterations)
            break;

        struct stat attr;
        if (!stat(args.infile.c_str(), &attr)) {
            unsigned long time = (unsigned long)attr.st_mtime;
            if (last_modified == 0)
                last_modified = time;
            else if (time != last_modified) {
                cout << "Updating assets! " << last_modified << " -> " << time
                     << endl;
                last_modified = time;
                vector<Float3> nv;
                geom.clear();
                load(args.infile, geom, 100, &mat_array[0]);
                copy(geom.begin(), geom.end(), geom_array);
                // TODO: fix lights
            }
        }

        // limit framerate
        auto t1 = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        cout << "\e[A\e[1G\e[0K"
             << "iteration " << iteration << ". " << (1000.f / dt) << "fps"
             << flush << endl;

        auto dtotal =
            chrono::duration_cast<chrono::milliseconds>(t1 - start_time)
                .count();
        cout << "\e[1G\e[0K"
             << "average time per frame: " << (dtotal / iteration) << flush;

        SDL_Delay(util::max(0l, 1000 / TARGET_FPS - dt));
    }

    // output rendered image
    if (args.output) {
        if (args.gpu) {
            // TODO: copy data back from GPU when in GPU mode
            assert(false);
        }
        output_bmp(pixels, w, h, args.outfile);
    }

    if (args.time) {
        auto end_time = chrono::high_resolution_clock::now();
        auto total_time =
            chrono::duration_cast<chrono::milliseconds>(end_time - start_time)
                .count();
        output_time(total_time, iteration, geom.size(),
                    PRIMARY_RAYS * args.width * args.height * iteration,
                    args.gpu, args.accel);
    }

    // GPU teardown
    if (args.gpu)
        cuda_destroy();

    // Once finished with OpenGL functions, the SDL_GLContext can be deleted.
    SDL_GL_DeleteContext(glcontext);
    sdl_check();

    util::hostdev_free(mat_array, args.gpu);
    util::hostdev_free(geom_array, args.gpu);
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
 * @param[in] texture_id the texture to bind to
 * @return the new buffer ID
 */
GLuint gl_create_buffer(int w, int h) {
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
void gl_buf2tex(int w, int h, GLuint buffer_id, GLuint texture_id) {
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
 * The pixel data must be in RGBA Float32 format.
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

/**
 * @brief Output a pixel array to a bitmap
 *
 * @param[in] pixels an RGBA Float32 array
 * @param[in] w width of the input array in pixels
 * @param[in] h height of the input array in pixels
 * @param[in] outfile path to output to
 */
void output_bmp(float *pixels, int w, int h, string outfile) {
    SDL_Surface *surf = SDL_CreateRGBSurface(0, w, h, 24, 0, 0, 0, 0);
    sdl_check();

    // copy pixels to surface
    uint8_t *spixels = static_cast<uint8_t *>(surf->pixels);
    for (int i = 0; i < w * h; ++i) {
        spixels[i * 3 + 2] = color_float_to_byte(pixels[i * 4 + 0]);
        spixels[i * 3 + 1] = color_float_to_byte(pixels[i * 4 + 1]);
        spixels[i * 3 + 0] = color_float_to_byte(pixels[i * 4 + 2]);
    }

    // write bitmap
    SDL_SaveBMP(surf, outfile.c_str());
    sdl_check();

    SDL_FreeSurface(surf);
    sdl_check();
}

/**
 * @brief Output timing data in csv format
 *
 * @param[in] total_time total time elapsed
 * @param[in] iteration total number of iterations
 * @param[in] geom_size number of geometries that are being tracing in the scene
 * @param[in] rays_cast total number of primary rays cast
 * @param[in] gpu whether or not the gpu is being used in tracing
 * @param[in] accel whether or not the acceleration structures are being used in
 * tracing
 */
void output_time(double total_time, unsigned iteration, size_t geom_size,
                 unsigned long rays_cast, bool gpu, bool accel) {
    double avg_time = total_time / iteration;

    cout << gpu << endl;
    cout << accel << endl;
    cout << total_time << endl;
    cout << iteration << endl;
    cout << avg_time << endl;
    cout << geom_size << endl;
    cout << rays_cast << endl;
}
