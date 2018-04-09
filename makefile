SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LDFLAGS := $(shell sdl2-config --libs)

CXX = g++
CXXFLAGS = -Wall -std=c++11 $(SDL_CFLAGS) -pthread -Ofast
LDFLAGS = $(SDL_LDFLAGS)
LDLIBS = -lGL -lGLEW

TRACER_SRC = tracer.c++ render.c++ Vec3.hh Sphere.hh
TRACER_OBJ = $(TRACER_SRC:%.c++=%.o)

all: tracer unit-tests

clean:
	-rm -f tracer
	-rm -f *.o

format:
	clang-format -i *.c++ *.hh

tracer: $(TRACER_OBJ) 
	$(CXX) $^ -o tracer $(LDFLAGS) $(LDLIBS)

test_tracer: tracer test_tracer.c++
	$(CXX) test_tracer.c++ -o test_tracer $(LDFLAGS) $(LDLIBS) -lgtest -lgtest_main -pthread

%.o: %.c++ $(wildcard %.hh)
	$(CXX) $(CXXFLAGS) $^ -c

unit-tests: test_tracer
	./test_tracer