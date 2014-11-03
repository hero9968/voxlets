
shoebox_helpers.py is a file with some functions used in all the voxlet stuff

------------------------------------
## For meshes:

1)  shoevoxes_from_bb_mesh.py
    to extract a load of shoeboxes from the meshes of the bigbird dataset

2)  create_voxlet_dict.py
    to load all these shoeboxes and cluster them
    This file can do either just training set or all - so be careful!

3)  save_centres_to_vdb.py
    To convert the found cluster centers to openvdb format

------------------------------------
## For images:

1)  extract_all_image_shoeboxes.py
    To get shoeboxes and features from all the images

2)  combine_sboxes_from_images.py
    To combine toether all the separate training shoebox files... to allow for dict to be formed

3)  create_dict_from_images.py
    To do the kmeans clustering on the combined sboxes

4)  classify_voxlets_against_dict.py
    Goes through the training (and test?) voxlets and sees which dictionary cluster they belong in. Combines into one big file