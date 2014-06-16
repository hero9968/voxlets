Marginalisation
===============

## For single object, single segmentation.

Image is $I$, event of occupancy is $o$. Basis shape is $B$.
\\[
P(o | I) = \sum_i P(o|B_i)P(B_i|I)
\\] 
We know that $P(o|B_i)$ is just 1 within the shape, and 0 outside. $P(B_i|I)$ is harder but could be based on some kind of heuristic based on number of inliers and how well the shape matches in etc.

In particular:
\\[
\sum_i P(B_i|I) = 1
\\]

## For multiple segmentations.

Now we have to introduce the concept of segmentations. The image $I$ will be segmented into a number of different regions $R$, where these regions may (but need not) overlap and the union of the regions need not cover the whole image.

We can then perform the fitting as before, but this time on a per-region basis. The regions are then marginalised out:
\\[
P(o | I) = \sum_{i,j} P(o|B_i)P(B_i|R_j)P(R_j|I)
\\]

Similar to before,
\\[
\sum_i P(B_i|R) = 1
\\]

However, there is no obligation for $\sum_j P(R_j|I) = 1$, as the regions may be non-overlapping.

The only thing left to compute is the probability of a specific region given the image, i.e. $P(R_j|I)$. 

\\[
P(R_j | I) = \sum_{k} P(R_j|S_k)P(S_k|I)
\\]
$P(R_j|S_k)$ is just the probability 