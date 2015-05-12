# class to set up the parameters for the system...
import socket

host_name = socket.gethostname()

# note: fewer testing cores as it tends to be more memory-intensive...
if False:
    # host_name == 'troll' or host_name == 'biryani':
    small_sample = False
    max_sequences = 500
    cores = 8
    testing_cores = 3
    multicore = True
else:
    small_sample = True
    max_sequences = 20
    cores = 8
    testing_cores = 3
    multicore = False
