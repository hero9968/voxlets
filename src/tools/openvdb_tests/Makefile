CC=clang++
CFLAGS=-I/Users/Michael/builds/openvdb_etc/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb

all: hellomake create yaml

hellomake: hello_world.cpp
	$(CC) -o hellomake hello_world.cpp $(CFLAGS)

create: create_write_grid.cpp
	$(CC) -o create create_write_grid.cpp $(CFLAGS)

yaml: yaml_parser_test.cpp
	$(CC) -o yaml yaml_parser_test.cpp  -lyaml-cpp $(CFLAGS)

clean:
	rm -rf *o hello