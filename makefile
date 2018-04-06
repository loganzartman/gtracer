SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LDFLAGS := $(shell sdl2-config --libs)

CXX = g++
CXXFLAGS = -Wall $(SDL_CFLAGS)
LDFLAGS = $(SDL_LDFLAGS)
LDLIBS = -lGL

TRACER_SRC = tracer.c++
TRACER_OBJ = $(TRACER_SRC:%.c++=%.o)

all: tracer

clean:
	-rm -f tracer
	-rm -f *.o

format:
	clang-format -i *.c++ *.h

tracer: $(TRACER_OBJ) 
	$(CXX) $(LDFLAGS) tracer.o -o tracer $(LDLIBS)

%.o: %.c++ $(wildcard %.h)
	$(CXX) $(CXXFLAGS) $^ -c
