SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LDFLAGS := $(shell sdl2-config --libs)

CXX = g++
CXXFLAGS = -Wall $(SDL_CFLAGS)
LDFLAGS = $(SDL_LDFLAGS)
LDLIBS = -lGL

all: tracer

clean:
	-rm -f tracer
	-rm -f *.o

format:
	clang-format -i *.c++ *.h

tracer: tracer.o
	$(CXX) $(LDFLAGS) tracer.o -o tracer $(LDLIBS)

%.o: %.c++ $(wildcard %.h)
	$(CXX) $(CXXFLAGS) $^ -c