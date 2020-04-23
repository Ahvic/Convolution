CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`

convolution: convolution.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
