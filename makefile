SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LDFLAGS := $(shell sdl2-config --libs)

CXX = g++
CXXFLAGS = -Wall -std=c++11 $(SDL_CFLAGS) -pthread -Ofast
LDFLAGS = $(SDL_LDFLAGS)
LDLIBS = -lGL -lGLEW

TRACER_SRC = tracer.c++ render.c++
TRACER_HH  = Vec3.hh Mat.hh transform.hh Sphere.hh transform.hh
TRACER_OBJ = $(TRACER_SRC:%.c++=%.o)

all: tracer

clean:
	-rm -f tracer
	-rm -f *.o

format:
	clang-format -i *.c++ *.hh

tracer: $(TRACER_OBJ) $(TRACER_HH)
	$(CXX) $(CXXFLAGS) $^ -o tracer $(LDFLAGS) $(LDLIBS)

test_tracer: tracer test_tracer.c++
	$(CXX) $(CXXFLAGS) $(TRACER_HH) render.o test_tracer.c++ -o test_tracer $(LDFLAGS) $(LDLIBS) -lgtest -lgtest_main -pthread

%.o: %.c++ $(wildcard %.hh)
	$(CXX) $(CXXFLAGS) $^ -c

unit-tests: test_tracer
	./test_tracer
