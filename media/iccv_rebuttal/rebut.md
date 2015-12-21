
Rebuttal

Thanks to R10, R18, and R20 …

We are delighted that R10 thinks that "This is a great paper, which [they] would like to see in ICCV."

We note that we are the first paper to quantitatively evaluate the effects of voxel completion on datasets with ground truth. As noted in the reviews, our approach "strongly outperforms the state-of-the-art from Zheng et al".

Some of the reviews offered some opportunities for us to clarify our approach and demonstrate further introspection into its capabilities. We outline these here:


Datasets

In our paper we explicitly target desktop scenes, due to their ubiquity in the real world. We note that training and testing on tabletop scenes follows recent state-of-the-art approaches in object pose detection, e.g. "Learning 6D Object Pose Estimation using 3D Object Coordinates" from ECCV 2014.

Reviewer 10 asks "were the same 3D models present in the training and test sets...?".
To clarify this:

- The tabletop dataset contains a comprehensive train/test split at the object level. We have made this explicit in the supplementary material. In fact, each split of the dataset was captured using objects gathered at a different physical venue. This dataset is designed to show how our algorithm performs given a wide range of training objects.

- The synthetic dataset uses the same set of primitives in the training and test phase, but at a widly different randomly generated set of poses. This dataset demonstrates the performance that could be expected if our algorithm had training-time access to a comprehensive range of objects.



Reviewer 18 cites as a strength that there is no
Reviewers 18 and 20

Reviewer 18 - "Does the method is evaluated to reconstruct more complex shape other than those simple shape?":


Reviewer 20 suggests that "the results actually suggest that semantic and 2D appearance information are critical to achieve good performance when training data are limited". In our approach we show how much is possible without semantics, which previous attempts at similar problems (e.g. [20, 41] did not. Reviewer 20 recognises the benefit of this, noting that "the proposed method is potentially more general and can be trained with 3D data with no semantic annotations", and reviewer 18 notes that ". No appearance and semantic information are required" as a strength of the paper.



Experiements to do:

1. Different scales of voxlets
2. Zheng on tt and synth - quantitative
3. Quantity of training data on tabletop (graph)
4. Quantitative comparison to zheng on NYU,
5. Occlusion sweeping on tabletop:
    - simulated effects of occlusion on test scenes
    - swpet plane 30% 60%
    - results lwoered by x and y respecvtiverly
    - images in sup material
6. Best voxlet to choose from forest - include as a paragraph in methodology
7. Train synthetic, test tabletop (low priority)