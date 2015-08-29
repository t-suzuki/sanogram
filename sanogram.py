#!env python
# sanogram.py - convert image to a Tokyo Olympic Logo
# notice: limited for personal use. output image is a modification of the input image and
#         some extremely basic parts (like circles, fans, boxes) and users should be aware 
#         of copyrights of those images.
#
# License: Public Domain

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# patch elements.
def hex2rgbfloat(hexstr):
    hexstr = hexstr.replace('#', '')
    return [int(hexstr[0:2], base=16)/255.0, int(hexstr[2:4], base=16)/255.0, int(hexstr[4:6], base=16)/255.0]

COLOR_RED    = hex2rgbfloat('#DD2727')
COLOR_BLACK  = hex2rgbfloat('#2A2A2A')
COLOR_GOLD   = hex2rgbfloat('#A47F37')
COLOR_SILVER = hex2rgbfloat('#A5A5A5')
COLOR_BG     = hex2rgbfloat('#FFFFFF')

class SanoElement(object):
    def __init__(self, size_px, color, type_name, is_original):
        self.size_px = size_px
        self.color = color
        self.type_name = type_name
        self.patch = np.zeros((size_px, size_px, 3), dtype=np.float32)
        shape = SanoElement.create_shape(size_px, type_name, is_original)
        self.shape = shape
        self.patch[shape] = color
        self.patch[~shape] = COLOR_BG
    @classmethod
    def create_shape(cls, w, type_name, is_original):
        shape_patch = np.ndarray((w, w), dtype=np.bool)
        shape_patch[:, :] = False
        r2 = (w*3/2.0)**2.0 + (w/2.0)**2.0
        drawing = {
                (False, 'LT'): lambda x, y: (x + 0.5 - w*3/2.0)**2.0 + (y + 0.5 - w*3/2.0)**2.0 > r2,
                (False, 'RB'): lambda x, y: (x + 0.5 + w/2.0)**2.0 + (y + 0.5 + w/2.0)**2.0 > r2,
                (False, 'BX'): lambda x, y: True,
                (False, 'BG'): lambda x, y: False,
                (False, 'RN'): lambda x, y: (x + 0.5 - w/2.0)**2.0 + (y + 0.5 - w/2.0)**2.0 < (w/2.0)**2.0,
                }
        drawing.update({
                (True,  'LT'): lambda x, y: (w - x) > y,
                (True,  'RB'): lambda x, y: (w - x) < y,
                (True,  'BX'): drawing[(False, 'BX')],
                (True,  'BG'): drawing[(False, 'BG')],
                (True,  'RN'): drawing[(False, 'RN')],
                })
        f = drawing[(is_original, type_name)]
        for y in range(w):
            for x in range(w):
                shape_patch[y, x] = f(x, y)
        return shape_patch

class SanoElementSet(object):
    def __init__(self, size_px, is_original):
        e_LT = SanoElement(size_px, COLOR_GOLD, 'LT', is_original)
        e_RB = SanoElement(size_px, COLOR_SILVER, 'RB', is_original)
        e_BX = SanoElement(size_px, COLOR_BLACK, 'BX', is_original)
        e_RN = SanoElement(size_px, COLOR_RED, 'RN', is_original)
        e_BG = SanoElement(size_px, COLOR_BG, 'BG', is_original)
        self.elements = [e_LT, e_RB, e_BX, e_RN, e_BG]
        self.patches = [e.patch for e in self.elements]
        self.shapes = [e.shape for e in self.elements]

# --------------------------------------------------------------------------------
# catalogue of error function to find optimal patch.

def error_color(patch, elem):
    return np.sum(np.abs(patch - elem.patch))

def error_gray(patch, elem):
    return np.sum(np.abs(patch.sum(axis=2) - elem.patch.sum(axis=2)))

ERROR_FUNCS = {
        'color': error_color,
        'gray': error_gray,
        }

# --------------------------------------------------------------------------------
def create_sanogram(img, grid_size, error_func):
    # block coordinates
    h, w = img.shape[:2]
    blocks = []
    for iy in range(h/grid_size):
        for ix in range(w/grid_size):
            by, bx = iy*grid_size, ix*grid_size
            ey, ex = min(h, by+grid_size), min(w, bx+grid_size)
            blocks.append((iy, ix, by, bx, ey, ex))
    bh, bw = iy+1, ix+1
    # init labels unassigned.
    labels = np.ndarray((bh, bw), dtype=np.int32)
    labels[:, :] = -1
    # elements
    sano = SanoElementSet(grid_size, False)
    # find best patches
    h, w = img.shape[:2]
    for iy, ix, by, bx, ey, ex in blocks:
        if (ey-by < grid_size) or (ex-bx < grid_size): continue
        patch = img[by:ey, bx:ex]
        errors = [error_func(patch, elem) for i, elem in enumerate(sano.elements)]
        min_idx = np.argmin(errors)
        labels[iy, ix] = min_idx
    # apply labels
    res_img = np.zeros_like(img) + COLOR_BG
    for iy, ix, by, bx, ey, ex in blocks:
        if (ey-by < grid_size) or (ex-bx < grid_size): continue
        label = labels[iy, ix]
        if label >= 0:
            res_img[by:ey, bx:ex] = sano.elements[label].patch
    return res_img

def show_elements_test(is_original):
    sano = SanoElementSet(W, False)
    fig, axs = plt.subplots(2, 3)
    axs[0][0].imshow(sano.patches[0])
    axs[0][1].imshow(sano.patches[1])
    axs[0][2].imshow(sano.patches[2])
    axs[1][0].imshow(sano.patches[3])
    axs[1][1].imshow(sano.patches[4])

if __name__=='__main__':
    import sys
    import os
    import skimage.io
    fn = sys.argv[1]
    W = int(sys.argv[2])

    img = skimage.io.imread(fn)
    print 'Input Image: %dx%d, Block: %d, Filename: %s' % (img.shape[1], img.shape[0], W, os.path.basename(fn))
    res = create_sanogram(img, W, ERROR_FUNCS['gray'])
    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Input Image: %dx%d, Block: %d,\nFilename: %s' % (img.shape[1], img.shape[0], W, os.path.basename(fn)))
    axs[0].imshow(img)
    axs[1].imshow(res)

    plt.show()

