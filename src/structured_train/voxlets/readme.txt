
Run:
shoevoxes_from_bb.py
to extract a load of shoeboxes from the bigbird dataset

Then:
create_voxlet_dict.py
to load all these shoeboxes and cluster them
This file can do either just training set or all - so be careful!

Then
save_centres_to_vdb.py
To convert the found cluster centers to openvdb format

Next, it goes to the notebooks really.
