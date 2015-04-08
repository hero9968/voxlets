How to run Infinitam_Alt with the python wrapper
================================================

1) Sensor must be plugged into the USB 2.0 port - I don't think it works in the USB 3.0 ports

2) Commands to capture data:

cd ~/builds/InfiniTAM_Alt/python_wrapper/
python run_infinitam.py

Wait for program to start up... Then the keystrokes are listed at the bottom of the screen. Key ones are:

s - start capturing data
e - pull data from GPU and exit
Q - quit without saving the voxels or the frames

3) If the tracking is lost and cannot be recovered, or the voxels become corrupted, there is no fix and no undo :(
Just exit with 'Q' and start again.

4) On-screen keyboard: You can get this by typing 'onboard &' at the terminal.

4) The python wrapper should take care of all the folder creation and moving the files around the computer after the program is run. Files saved to disk are:

** DURING CAPTURE (in C++) **keybosrd
/tmp/log.txt    is updated with frame number, timestamp and camera pose
/tmp/frames/  is populated with the RGB and depth frames

** AFTER CAPTURE (in C++) **
/tmp/dump.voxels   is filled with the TSDF blocks extacted from the GPU and RAM

** AFTER CAPTURE (in Python) **
New folder ./data/saved_xxxxx/ is created
/tmp/log.txt, /tmp/frames/ and /tmp/dump.voxels are all moved to ./data/saved_xxxxx/
./data/saved_xxxxx/dump.voxels is converted to mesh, saved in ./data/saved_xxxxx/dump.obj

5) Each frame is saved in very wasteful ppm and pgm formats. Each RGB frame is approx 1MB, each depth image approx 0.6MB. I am therefore only saving every 10th frame. On this laptop (the foldy one), there are approx 430 GB free on Mon 9th March 2015.  This should be enough for Hopefully this will be enough space for capture!

