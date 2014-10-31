
Run:
shoevoxes_from_bb_mesh.py
to extract a load of shoeboxes from the meshes of the bigbird dataset

Then:
create_voxlet_dict.py
to load all these shoeboxes and cluster them
This file can do either just training set or all - so be careful!

Then
save_centres_to_vdb.py
To convert the found cluster centers to openvdb format

It sort of gets to the notebooks at this stage (esp dictionary_lookup)
BUT:

Next:
extract_training_shoeboes.py
To get training data from the images and voxel data for each object...
This might have to be run on troll or something!
