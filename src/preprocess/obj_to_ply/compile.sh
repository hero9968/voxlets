cc -c -I. obj_to_ply.c 
cc obj_to_ply.o ply_io.o -lm
mv a.out /usr/local/bin/obj_to_ply