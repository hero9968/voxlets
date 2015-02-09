'''
functions for dealing with 2d canvases
'''
import numpy as np
import scipy
from scipy.ndimage.morphology import distance_transform_edt

# Set some parameters here
box_size = (20, 30)
circle_size = (30, 30)
canvas_size = (100, 100)


class Canvas(object):
    def __init__(self):
        pass

    def _random_rotated_box(self):
        box = np.ones(box_size, dtype=np.float32)#[0], box_size[1])
        pad_size = (max(box_size)/2, max(box_size)/2)
        box = np.pad(box, [pad_size, pad_size], mode='constant')
        return scipy.misc.imrotate(
            box, np.random.rand() * 360, interp='nearest').astype(np.float32) / 255


    def random_fill(self):
        '''
        fill with random scene
        '''
        circle = 255 - scipy.misc.imread('data/circle.png')
        circle = scipy.misc.imresize(
            circle, circle_size, interp='nearest').astype(np.float32) / 255

        box = self._random_rotated_box()

        canvas = np.zeros(canvas_size)

        box_loc = (np.random.randint(0, canvas.shape[0] - box.shape[0]),
                   np.random.randint(0, canvas.shape[1] - box.shape[1]))

        circle_loc = (np.random.randint(0, canvas.shape[0] - circle.shape[0]),
                      np.random.randint(0, canvas.shape[1] - circle.shape[1]))

        canvas[box_loc[0]:(box_loc[0] + box.shape[0]),
               box_loc[1]:(box_loc[1] + box.shape[1])] += box
        canvas[circle_loc[0]:(circle_loc[0] + circle.shape[0]),
               circle_loc[1]:(circle_loc[1] + circle.shape[1])] += circle

        self.im = canvas > 0.5


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

        return sdf

    @classmethod
    def fill_and_render(cls):
        im = cls()
        im.random_fill()
        im.render_canvas()
        im.tsdf_from_render(10)
        return im
