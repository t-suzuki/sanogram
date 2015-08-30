#!env python
# sanogram.py - convert image to a Tokyo Olympic Logo
# notice: limited for personal use. output image is a modification of the input image and
#         some extremely basic parts (like circles, fans, boxes), and users should be aware 
#         of copyrights of those images. use at your own risk.
#
# License: Public Domain

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster

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
    def __init__(self, size_px, color, color_bg, type_name, is_original):
        self.size_px = size_px
        self.color = color
        self.color_bg = color_bg
        self.type_name = type_name
        self.patch = np.zeros((size_px, size_px, 3), dtype=np.float32)
        shape = SanoElement.create_shape(size_px, type_name, is_original)
        self.shape = shape
        self.patch[shape] = color
        self.patch[~shape] = color_bg
        self.is_background = (type_name == 'BG')
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
        e_LT = SanoElement(size_px, COLOR_GOLD, COLOR_BG, 'LT', is_original)
        e_RB = SanoElement(size_px, COLOR_SILVER, COLOR_BG, 'RB', is_original)
        e_BX = SanoElement(size_px, COLOR_BLACK, COLOR_BG, 'BX', is_original)
        e_RN = SanoElement(size_px, COLOR_RED, COLOR_BG, 'RN', is_original)
        e_BG = SanoElement(size_px, COLOR_BG, COLOR_BG, 'BG', is_original)
        self.elements = [e_LT, e_RB, e_BX, e_RN, e_BG]
        self.patches = [e.patch for e in self.elements]
        self.shapes = [e.shape for e in self.elements]
        self.background_color = COLOR_BG
        self.block_px = size_px

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
def create_sanogram(elements_set, img, error_func, replace_color=None, n_colors=5):
    grid_size = elements_set.block_px
    # block coordinates
    h, w = img.shape[:2]
    blocks = []
    for iy in range(h/grid_size):
        for ix in range(w/grid_size):
            by, bx = iy*grid_size, ix*grid_size
            ey, ex = min(h, by+grid_size), min(w, bx+grid_size)
            if (ey-by < grid_size) or (ex-bx < grid_size): continue
            patch = img[by:ey, bx:ex]
            blocks.append((iy, ix, by, bx, ey, ex, patch))
    bh, bw = iy+1, ix+1
    # init labels unassigned.
    labels = np.ndarray((bh, bw), dtype=np.int32)
    labels[:, :] = -1
    # find best patches
    h, w = img.shape[:2]
    for iy, ix, by, bx, ey, ex, patch in blocks:
        errors = [error_func(patch, elem) for i, elem in enumerate(elements_set.elements)]
        min_idx = np.argmin(errors)
        labels[iy, ix] = min_idx
    # determine the new color
    if replace_color == 'direct':
        # use mean color of the target patch directly.
        color_map = np.ndarray((bh, bw, 3), dtype=img.dtype)
        for iy, ix, by, bx, ey, ex, patch in blocks:
            label = labels[iy, ix]
            if not elements_set.elements[label].is_background:
                mean_color = patch[elements_set.elements[label].shape].mean(axis=0)
                color_map[iy, ix] = mean_color
    elif replace_color == 'representative':
        # find <n_colors> representative colors from the input image and use the nearest one for each patch.
        colors = img.reshape((-1, 3))
        cluster = sklearn.cluster.KMeans(n_clusters=n_colors)
        cluster.fit(colors)
        # assign colors
        color_map = np.ndarray((bh, bw, 3), dtype=img.dtype)
        for iy, ix, by, bx, ey, ex, patch in blocks:
            label = labels[iy, ix]
            if not elements_set.elements[label].is_background:
                representative_index = cluster.predict((patch[elements_set.elements[label].shape]).mean(axis=0))
                color_map[iy, ix] = cluster.cluster_centers_[representative_index]
    elif replace_color is None:
        # color is associated to the patch shape according to elements_set.
        color_map = None
    else:
        color_map = None
        print 'unknown replace_color=%s' % replace_color

    # apply labels
    res_img = np.zeros_like(img) + COLOR_BG
    for iy, ix, by, bx, ey, ex, patch in blocks:
        label = labels[iy, ix]
        if label >= 0:
            if color_map is None:
                res_img[by:ey, bx:ex] = elements_set.elements[label].patch
            else:
                res_img[by:ey, bx:ex][elements_set.elements[label].shape] = color_map[iy, ix]
                res_img[by:ey, bx:ex][~elements_set.elements[label].shape] = elements_set.background_color
    return res_img

def show_elements_test(is_original):
    W = 32
    sano = SanoElementSet(W, is_original)
    fig, axs = plt.subplots(2, 3)
    axs[0][0].imshow(sano.patches[0])
    axs[0][1].imshow(sano.patches[1])
    axs[0][2].imshow(sano.patches[2])
    axs[1][0].imshow(sano.patches[3])
    axs[1][1].imshow(sano.patches[4])

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='sanogram.py')
    parser.add_argument('--input', type=str, required=True,
            help='input file name')
    parser.add_argument('--block', type=int, default=None,
            help='block size (in pixel)')
    parser.add_argument('--quiet', action='store_true',
            help='do not show the result')
    parser.add_argument('--write_dir', type=str, default=None,
            help='save the result to a directory')
    parser.add_argument('--original', action='store_true',
            help='use the "original"-like elements instead of "final"-like elements')
    parser.add_argument('--error', type=str, default='gray',
            help='error function used for optimization. [gray|color]')
    parser.add_argument('--replace_color', type=str, default=None,
            help='replace the color of elements from the input image. [direct|representative]')
    return parser.parse_args()

if __name__=='__main__':
    import sys
    import os
    import skimage.io
    args = parse_args()
    fn = args.input

    img = skimage.io.imread(fn)
    h, w = img.shape[:2]
    if args.block is None:
        block_px = max(8, min(w, h)/10)
    else:
        block_px = args.block

    elements = SanoElementSet(block_px, args.original)

    print 'Input Image: %dx%d, Block: %d, Filename: %s' % (w, h, block_px, os.path.basename(fn))
    res = create_sanogram(elements, img, ERROR_FUNCS[args.error], replace_color=args.replace_color, n_colors=5)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Sanogram of Size: %dx%d, Block: %d, Filename: %s' % (w, h, block_px, os.path.basename(fn)))
    axs[0].imshow(img)
    axs[1].imshow(res)
    fig.tight_layout()

    if args.write_dir is not None:
        base, ext = os.path.splitext(os.path.basename(fn))
        write_fn = os.path.join(args.write_dir, '%s_sanogram%s' % (base, ext))
        fig.savefig(write_fn)
        print 'wrote to:', write_fn

    if not args.quiet:
        plt.show()

