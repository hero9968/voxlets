# cc -c -I. obj_to_ply.c 
# cc obj_to_ply.o ply_io.o -lm
# mv a.out /usr/local/bin/obj_to_ply
clang -c obj_to_ply.c
clang -c ply_io.c
clang -o obj_to_ply obj_to_ply.o ply_io.o -lm
mv obj_to_ply /usr/local/bin/