SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LDFLAGS := $(shell sdl2-config --libs)

CXX = g++
CXXFLAGS = -Wall -std=c++11 $(SDL_CFLAGS) -g
LDFLAGS = $(SDL_LDFLAGS)
LDLIBS = -lGL -lGLEW

TRACER_SRC = tracer.c++ render.c++
TRACER_OBJ = $(TRACER_SRC:%.c++=%.o)

all: tracer

clean:
	-rm -f tracer
	-rm -f *.o

format:
	clang-format -i *.c++ *.hh

tracer: $(TRACER_OBJ) 
	$(CXX) $^ -o tracer $(LDFLAGS) $(LDLIBS)

%.o: %.c++ $(wildcard %.hh)
	$(CXX) $(CXXFLAGS) $^ -c
