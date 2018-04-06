#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>

#include "Sphere.h"
#include "Vec3.h"

#define TARGET_FPS 60

using namespace std;

int main() {
  // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
  SDL_Window *window =
      SDL_CreateWindow("SDL2/OpenGL Demo", 0, 0, 640, 480,
                       SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

  // Create an OpenGL context associated with the window.
  SDL_GLContext glcontext = SDL_GL_CreateContext(window);

  bool running = true;
  SDL_Event event;

  while (running) {
    auto t0 = chrono::high_resolution_clock::now();
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT)
        running = false;
    }

#define randf() (rand() % 10000 / 10000.)
    glClearColor(randf(), randf(), randf(), 1); // set clear color
    glClear(GL_COLOR_BUFFER_BIT);               // clear context
    SDL_GL_SwapWindow(window);                  // update window

    // limit framerate
    auto t1 = chrono::high_resolution_clock::now();
    auto dt = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    cout << dt << endl;
    SDL_Delay(max(0l, 1000 / TARGET_FPS - dt));
  }

  // Once finished with OpenGL functions, the SDL_GLContext can be deleted.
  SDL_GL_DeleteContext(glcontext);

  return 0;
}
