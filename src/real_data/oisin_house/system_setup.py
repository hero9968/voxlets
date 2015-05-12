# class to set up the parameters for the system...
import socket

host_name = socket.gethostname()

if False:
    # host_name == 'troll' or host_name == 'biryani':
    small_sample = False
    max_sequences = 500
    cores = 8
    multicore = True
else:
    small_sample = True
    max_sequences = 20
    cores = 8
    multicore = True
