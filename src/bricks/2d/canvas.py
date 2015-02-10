'''
functions for dealing with 2d canvases
'''
import numpy as np
import scipy
from scipy.ndimage.morphology import distance_transform_edt
from sklearn.neighbors import NearestNeighbors

# Set some parameters here
box_size = (5, 25)
circle_size = (15, 15)
canvas_size = (50, 50)
pad_region = 20

class Canvas(object):
    def __init__(self):
        pass

    def _random_rotated_box(self):
        box = np.ones(box_size, dtype=np.float32)#[0], box_size[1])
        pad_size = (max(box_size)/2, max(box_size)/2)
        box = np.pad(box, [pad_size, pad_size], mode='constant')
        return scipy.misc.imrotate(
            box, np.random.rand() * 360, interp='nearest').astype(np.float32) / 255

    def random_fill(self, num_boxes=1, num_circles=1):
        '''
        fill with random scene
        '''
        max_iters = 1000
        count = 0

        while True:
            circle = 255 - scipy.misc.imread('data/circle.png')
            circle = scipy.misc.imresize(
                circle, circle_size, interp='nearest').astype(np.float32) / 255

            canvas = np.zeros(canvas_size)

            for i in range(num_boxes):
                box = self._random_rotated_box()

                box_loc = (np.random.randint(0, canvas.shape[0] - box.shape[0]),
                           np.random.randint(0, canvas.shape[1] - box.shape[1]))

                canvas[box_loc[0]:(box_loc[0] + box.shape[0]),
                       box_loc[1]:(box_loc[1] + box.shape[1])] += box

            for i in range(num_circles):
                circle_loc = (np.random.randint(0, canvas.shape[0] - circle.shape[0]),
                              np.random.randint(0, canvas.shape[1] - circle.shape[1]))
                canvas[circle_loc[0]:(circle_loc[0] + circle.shape[0]),
                       circle_loc[1]:(circle_loc[1] + circle.shape[1])] += circle

            if np.max(canvas) < 1.0001:
                break
            elif count > max_iters:
                raise Exception("Couldn't generate a canvas")
            count += 1

        print count
        self.im = canvas > 0.5
        self.im = np.pad(
            self.im, [[pad_region, pad_region], [pad_region, pad_region]],
            'constant', constant_values=0)

    def render_canvas(self):
        '''renders canvas from the top down'''
        pix_in_col =  np.array([np.where(col==1)[0] for col in self.im.T])
        self.depth_image = \
            [pix[0] if pix.shape[0] > 0 else self.im.shape[0] for pix in pix_in_col]

    # maybe now do a tsdf on this?
    def tsdf_from_render(self, mu):
        ray_range = np.arange(self.im.shape[0])
        self.render_tsdf = np.zeros(self.im.shape)

        for depth, col in zip(self.depth_image, self.render_tsdf.T):

            if depth == self.im.shape[0]:
                col[:] = mu
                continue
            to_fill = ray_range - depth < mu
            col[to_fill] = depth - ray_range[to_fill]

        self.render_tsdf[self.render_tsdf > mu] = mu
        self.render_tsdf[self.render_tsdf < -mu] = -mu

        return self.render_tsdf

    def full_tsdf(self, mu):

        trans_inside = distance_transform_edt(self.im)
        trans_outside = distance_transform_edt(1-self.im)
        sdf = trans_outside - trans_inside

        # truncate
        sdf[sdf > mu] = mu
        sdf[sdf < -mu] = -mu

        self.sdf = sdf
        return self.sdf

    @classmethod
    def fill_and_render(cls, num_boxes=1, num_circles=1):
        im = cls()
        im.random_fill(num_boxes, num_circles)
        im.render_canvas()
        im.tsdf_from_render(10)
        im.full_tsdf(10)
        return im

    def extract_windows(self, sliding_window_shape):

        potential_places = np.array(
            [(point, idx)
            for idx, point in enumerate(self.depth_image)
            if point < self.im.shape[0]])

        all_X, all_Y = [], []

        for place in potential_places:

            top = place[0] - sliding_window_shape[0] / 2
            bottom = place[0] + sliding_window_shape[0] / 2
            left = place[1] - sliding_window_shape[1] / 2
            right = place[1] + sliding_window_shape[1] / 2

            Y = self.im[top:bottom, left:right]
            X = self.render_tsdf[top:bottom, left:right]

            all_X.append(X.flatten())
            all_Y.append(Y.flatten())

        all_X_np = np.vstack(all_X)
        all_Y_np = np.vstack(all_Y)

        return all_X_np, all_Y_np


    def predict_from_windows(
            self, sliding_window_shape, train_X, train_Y, patch_in_known=True):
        '''
        uses train_X and train_Y to try to fill in the missing parts!
        probably wont work too well but we'll see...
        '''

        # follwing line should perhaps not be here...
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(train_X)

        potential_places = np.array(
            [(point, idx)
            for idx, point in enumerate(self.depth_image)
            if point < self.im.shape[0]])

        prediction = np.zeros(self.im.shape)

        for place in potential_places:

            top = place[0] - sliding_window_shape[0] / 2
            bottom = place[0] + sliding_window_shape[0] / 2
            left = place[1] - sliding_window_shape[1] / 2
            right = place[1] + sliding_window_shape[1] / 2

            Y = self.im[top:bottom, left:right]
            X = self.render_tsdf[top:bottom, left:right]

            # finding a match for the X...
            distances, indices = nbrs.kneighbors(X.flatten())

            # patching the prediction into the prediction image...
            # Using the gt:
            #prediction[top:bottom, left:right] = Y
            # Using the machine learning:
            try:
                prediction[top:bottom, left:right] += \
                    train_Y[indices[0][0]].reshape(sliding_window_shape)
            except:
                print train_Y.shape
                print train_Y[indices[0][0]].shape
                print sliding_window_shape

            #print indices

        if patch_in_known:
            prediction[self.render_tsdf == self.render_tsdf.max()] = 0
            #prediction[self.render_tsdf < 0] = 1

        return prediction



# '''
# now extract sliding windows from all over the rendered tsdf
# want to then be able to say for any sliding window location input,
# to detect ('fire') where we see a good example of something we can predict for
# i.e. close to a training example
# '''
# sliding_window_shape = (20, 20)
# canvas = training_canvases[0]
# total = 0

# # Dense method
# training_windows = []
# for row_idx in range(canvas.shape[0] - sliding_window_shape[0]):
#     for col_idx in range(canvas.shape[1] - sliding_window_shape[1]):
#         temp = canvas[row_idx:row_idx + sliding_window_shape[0],
#                       col_idx:col_idx + sliding_window_shape[1]]
#         if np.any(temp):
#             training_windows.append(temp)
#         total += 1

# print total, len(training_windows)