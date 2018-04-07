#include "SDL.h"
#include "SDL_opengl.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "render.hh"
#include "Sphere.hh"
#include "Vec3.hh"

#define TARGET_FPS 60
#define SPHERES 5
#define WIDTH 640
#define HEIGHT 480 

using namespace std;

void output_to_ppm (float3 *image, size_t w, size_t h);
void sdl_check();


int main() {
    // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
    SDL_Window *window =
        SDL_CreateWindow("SDL2/OpenGL Demo", 0, 0, WIDTH, HEIGHT,
                         SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    sdl_check();

    // Create an OpenGL context associated with the window.
    SDL_GLContext glcontext = SDL_GL_CreateContext(window);
    sdl_check();

    bool running = true;
    SDL_Event event;

    Sphere s0(float3(5,5,10),      2,   float3(1,0,0));
    Sphere s1(float3(WIDTH/2,HEIGHT/2,20),   1.5, float3(1,0,0));
    Sphere s2(float3(-5,-1,15),   1,   float3(1,0,0));
    Sphere s3(float3(5,1,30),     1,   float3(1,0,0));

    // light source
    Sphere l0(float3(-10,2,10), 5,   float3(1,0,0), 0, 0, float3(3));

    Sphere spheres[SPHERES] = { s0, s1, s2, s3, l0 };

    float3 *image;

    while (running) {
        auto t0 = chrono::high_resolution_clock::now();
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
        }

        image = cpu_render(spheres, SPHERES, WIDTH, HEIGHT);
        output_to_ppm(image, WIDTH, HEIGHT);
        running = false;

#define randf() (rand() % 10000 / 10000.)
        glClearColor(randf(), randf(), randf(), 1);  // set clear color
        glClear(GL_COLOR_BUFFER_BIT);                // clear context
        SDL_GL_SwapWindow(window);                   // update window

        // limit framerate
        auto t1 = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        cout << dt << endl;
        SDL_Delay(max(0l, 1000 / TARGET_FPS - dt));
    }

    // Once finished with OpenGL functions, the SDL_GLContext can be deleted.
    SDL_GL_DeleteContext(glcontext);
    sdl_check();

    delete [] image; 

    return 0;
}

// Util

void output_to_ppm (float3 *image, size_t w, size_t h) {
  std::ofstream ofs("./output.ppm", std::ios::out | std::ios::binary); 
  ofs << "P6\n" << w << " " << h << "\n255\n"; 
  for (unsigned i = 0; i < w * h; ++i) { 
      ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) << 
             (unsigned char)(std::min(float(1), image[i].y) * 255) << 
             (unsigned char)(std::min(float(1), image[i].z) * 255); 
  } 
  ofs.close(); 
}

void sdl_check() {
    const char *err = SDL_GetError();
    if (err[0] != 0) {
        cout << "SDL Error:" << endl;
        cout << err << endl;
        SDL_ClearError();
    }
}
