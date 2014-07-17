CC=clang++
CFLAGS=-I/Users/Michael/builds/openvdb_etc/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb

all: hello create

hello: hello_world.cpp
	$(CC) -o hellomake hello_world.cpp $(CFLAGS)

create: create_write_grid.cpp
	$(CC) -o create create_write_grid.cpp $(CFLAGS)

clean:
	rm -rf *o hello