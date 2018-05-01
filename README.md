# gtracer
*GPU-accelerated path tracing*

Austin Atchley, Logan Zartman

## Dependencies
* SDL2
* GLEW
* CUDA
* Google Test

## Building
* `make`
* `make unit-tests`
* `export SDL_VIDEO_X11_VISUALID=` if you have issues relating to "Couldn't find matching GLX visual"
* `./tracer`

## Use
To view flags, run the binary with the `-?` flag

Give the tracer an object file as an argument.

After running, you can use the mouse and arrow keys to rotate and translate the camera.

If you specify an output location, the tracer will save a `.bmp` to that path after you exit the program

As the tracer runs, it will give FPS, iterations, and average milliseconds per frame counters. For each iteration, the tracer averages each image together, and the render becomes more clear over time. This is due to the randomness of path tracing (as opposed to ray tracing).
