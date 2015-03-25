fig = plt.figure(figsize=(10, 10))

ms.from_volume(gt_vox)

height = (ms.vertices[:, 2] > 0.1) & (ms.vertices[:, 2] < 0.2) & \
     (ms.vertices[:, 1] > 0.6)
plt.plot(ms.vertices[height, 0], ms.vertices[height, 1], '.')
plt.axis('equal');







# need to form a graph between neighbouring vertices using the faces
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix, lil_matrix

A = ms.faces[:, 0]
B = ms.faces[:, 1]
C = ms.faces[:, 2]

ABC = np.hstack((A, B, C, B, C, A))
BCA = np.hstack((B, C, A, A, B, C))
graph = coo_matrix((np.ones(ABC.shape[0]), (ABC, BCA)))
print graph.nnz, graph.shape






# now want to check the normals...
fig = plt.figure(figsize=(10, 10))
ms.compute_vertex_normals()

height = (ms.vertices[:, 2] > 0.1) & (ms.vertices[:, 2] < 0.2) & \
     (ms.vertices[:, 1] > 0.6)
plt.plot(ms.vertices[height, 0], ms.vertices[height, 1], '.')
for idx in np.where(height)[0]:
    X = ms.vertices[idx, 0]
    Y = ms.vertices[idx, 1]
    X1 = ms.vertices[idx, 0] + ms.norms[idx, 0] * 0.1
    Y1 = ms.vertices[idx, 1] + ms.norms[idx, 1] * 0.1
    plt.plot((X, X1), (Y, Y1))
plt.axis('equal');





import scipy

# Computing vertex normals
ms.norms = np.zeros(ms.vertices.shape)
# convert graph to different graph type
graph_lil = graph.tolil()

for idx, vert in enumerate(ms.vertices):

    neighbours = graph_lil.rows[idx]
    neighbours_xyz = ms.vertices[neighbours, :]

    if neighbours_xyz.shape[0] > 3:
        # take final column as the normal direction
        [u, d, v] = np.linalg.svd(neighbours_xyz,0)
        #print v.shape
        B = v[2,:];                    # Solution is last column of v.
        nn = np.linalg.norm(B)
        B = B / nn
    #    print
    #    beak
        norm = scipy.linalg.svd(neighbours_xyz)[0][-1, :]
        ms.norms[idx, :] = B[0:3]