CC=clang++
VDBFLAGS=-I/usr/local//OpenVDB/include/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb

all: combine_voxlets
#voxelvdb

combine_voxlets: combine_voxlets.cpp
	$(CC) -o combine_voxlets combine_voxlets.cpp -L/usr/local/lib -lyaml-cpp $(VDBFLAGS)

clean:
	rm -rf *o combine_voxlets
