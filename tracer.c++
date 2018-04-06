#include "SDL.h"
#include "SDL_opengl.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>

#include "render.hh"
#include "Sphere.hh"
#include "Vec3.hh"

#define TARGET_FPS 60
#define SPHERES 5

using namespace std;

void sdl_check() {
    const char *err = SDL_GetError();
    if (err[0] != 0) {
        cout << "SDL Error:" << endl;
        cout << err << endl;
        SDL_ClearError();
    }
}

int main() {
    // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
    SDL_Window *window =
        SDL_CreateWindow("SDL2/OpenGL Demo", 0, 0, 640, 480,
                         SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    sdl_check();

    // Create an OpenGL context associated with the window.
    SDL_GLContext glcontext = SDL_GL_CreateContext(window);
    sdl_check();

    bool running = true;
    SDL_Event event;

    Sphere s0(float3(1,1,3),    2, float3(1,0,0));
    Sphere s1(float3(10,0,2),   1.5, float3(1,0,0));
    Sphere s2(float3(0,-1,5),   1, float3(1,0,0));
    Sphere s3(float3(5,1,5),    1, float3(1,0,0));
    Sphere s4(float3(-10,2,10), 5, float3(1,0,0));

    Sphere spheres[SPHERES] = { s0, s1, s2, s3, s4 };

    while (running) {
        auto t0 = chrono::high_resolution_clock::now();
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
        }

        cpu_render(spheres, SPHERES);

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

    return 0;
}
