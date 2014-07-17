CC=clang++
#CFLAGS=-I/Users/Michael/builds/openvdb_etc/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb
CFLAGS=-I/usr/local//OpenVDB/include/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb

all: txt2vdb

txt2vdb: txt2vdb.cpp
	$(CC) -o txt2vdb txt2vdb.cpp $(CFLAGS)

clean:
	rm -rf *o txt2vdb