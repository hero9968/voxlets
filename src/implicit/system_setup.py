# class to set up the parameters for the system...
import socket

host_name = socket.gethostname()

# note: can have fewer testing cores in case it is more memory-intensive...
small_sample = True
max_sequences = 500
max_test_sequences = 40
cores = 3
testing_cores = 3
multicore = False
