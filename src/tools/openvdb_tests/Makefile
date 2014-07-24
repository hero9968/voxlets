CC=clang++
CFLAGS=-I/usr/local//OpenVDB/include/ -L/usr/local/lib/ -ltbb -lHalf  -lopenvdb

all: hellomake create yaml

hellomake: hello_world.cpp
	$(CC) -o $@ hello_world.cpp $(CFLAGS)

create: create_write_grid.cpp
	$(CC) -o $@ create_write_grid.cpp $(CFLAGS)

yaml: yaml_parser_test.cpp
	$(CC) -o $@ yaml_parser_test.cpp  -lyaml-cpp $(CFLAGS)

clean:
	rm -rf *.o hellomake create yaml