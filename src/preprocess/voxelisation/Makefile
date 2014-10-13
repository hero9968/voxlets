CC=clang++
#CFLAGS=-I/Users/Michael/builds/openvdb_etc/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb
VDBFLAGS=-I/usr/local//OpenVDB/include/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb

all: voxelcvml 
#voxelvdb

voxelcvml: voxelcvml.cpp
	$(CC) -o voxelcvml voxelcvml.cpp -lcvmlcpp
	
voxelvdb: voxelvdb.cpp
	$(CC) -o voxelvdb voxelvdb.cpp $(VDBFLAGS)

clean:
	rm -rf *o voxelcvml
	rm -rf *o voxelvdb
