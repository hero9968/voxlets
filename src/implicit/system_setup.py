# class to set up the parameters for the system...
import socket

host_name = socket.gethostname()

# note: can have fewer testing cores in case it is more memory-intensive...
small_sample = True
max_sequences = 50
max_test_sequences = 24
cores = 8
testing_cores = 8
multicore = True
