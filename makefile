.SUFFIXES: # disable builtin rules
SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LDFLAGS := $(shell sdl2-config --libs)

# feature detection
ifneq ($(wildcard /opt/cuda-8.0/.),)
	CUDADIR = /opt/cuda-8.0/lib64  # lab machines
else
	CUDADIR = /opt/cuda/lib64      # alternative
endif

ifneq ($(shell which python3),)
	PY = python3 # lab machines
else
	PY = python  # alternative
endif

# c++ variables
CXX = g++
override CXXFLAGS += -Wall -Wextra -Wno-unused-function -std=c++11 $(SDL_CFLAGS) -pthread $(OPTIM)
OPTIM = -Ofast
LDFLAGS = $(SDL_LDFLAGS) -L$(CUDADIR)
LDLIBS = -lGL -lGLEW -lcuda -lcudart
TRACER_SRC = tracer.c++ render.c++ loader.c++
TRACER_HH  = util.hh Vec3.hh Mat.hh transform.hh AABB.hh Tri.hh Geometry.hh Sphere.hh Box.hh UniformGrid.hh transform.hh util.hh raytracing.hh cuda_render.hh
TRACER_OBJ = $(TRACER_SRC:%.c++=%.o)

# cuda variables
NVCC     = nvcc
CUDA_SRC = cuda_util.cu cuda_render.cu
CUDA_HH  = cuda_util.hh cuda_util.cuh cuda_render.cuh
CUDA_OBJ = $(CUDA_SRC:%.cu=%.cu.o)
NVFLAGS  = -std=c++11 -arch=sm_52 -Xcompiler="-fPIC" -O3

all: tracer

clean:
	-rm -f tracer
	-rm -f test_tracer
	-rm -f *.o
	-rm -f *.gcda

format:
	clang-format -i *.c++ *.hh *.cu *.cuh

# link c++ and cuda objects into tracer executable
tracer: $(TRACER_SRC) $(TRACER_OBJ) $(CUDA_OBJ) $(TRACER_HH) $(CUDA_HH)
	$(CXX) $(CXXFLAGS) $(TRACER_OBJ) $(CUDA_OBJ) -o tracer $(LDFLAGS) $(LDLIBS)

# todo: support unit testing CUDA
test_tracer: test_tracer.c++ $(TRACER_SRC) $(TRACER_HH)
	$(CXX) $(filter-out tracer.c++,$(TRACER_SRC)) test_tracer.c++ -o test_tracer $(LDFLAGS) $(LDLIBS) -lgtest -lgtest_main -pthread

# c++ objects
%.o: %.c++ $(wildcard %.hh)
	$(CXX) $(CXXFLAGS) $< -c

# cuda objects
%.cu.o: %.cu $(wildcard %.cuh)
	$(NVCC) $(NVFLAGS) $< -c -o $@

unit-tests: test_tracer
	./test_tracer
