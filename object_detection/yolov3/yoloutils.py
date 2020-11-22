to_import = [
    'definitions'
    ]

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding
    '''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (255,255,255))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(
        annotation_line, input_shape, random=True, max_boxes=20,
        proc_img=True):
    '''random preprocessing for real-time data augmentation
    '''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    
    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BILINEAR)
            new_image = Image.new('RGB', (w,h), (255,255,255))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data
    
    # Rescale
    scale = rand(definitions.AugmentationParameters.scale_low, definitions.AugmentationParameters.scale_high) # old: .7, 1.3
    nh = int(scale*h)
    nw = int(scale*w)
    image = image.resize((nw,nh), Image.BILINEAR)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (255,255,255))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = False#rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Rotate or not
    rotate_90 = (rand()< definitions.AugmentationParameters.rotation_p) and definitions.AugmentationParameters.enable_90rotation
    rotate_180 = (rand()< definitions.AugmentationParameters.rotation_p) and definitions.AugmentationParameters.enable_180rotation
    if rotate_90: image = image.transpose(Image.ROTATE_90)
    if rotate_180: image = image.transpose(Image.ROTATE_180)
        
    # distort image
    """hue = 0 #rand(-hue, hue)
    sat = 1 #rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0"""
    #image_data = hsv_to_rgb(x) # numpy array, 0 to 1
    image_data = np.array(image)/255.
    
    # BOX: x1, y1, x2, y2
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        
        
        if rotate_90:
            box[:, [0,1,2,3]] = box[:, [1,2,3,0]] # Added
            box[:, [1,3]] = h - box[:, [1,3]]
            
        if rotate_180:
            # Fix this if w!=h
            box[:, [0,1,2,3]] = box[:, [1,2,3,0]] # Added
            box[:, [1,3]] = h - box[:, [1,3]]
            box[:, [0,1,2,3]] = box[:, [1,2,3,0]] # Added
            box[:, [1,3]] = h - box[:, [1,3]]
            
            #box[:, [0,1,2,3]] = box[:, [1,0,3,2]] # Added
        
        box_data[:len(box)] = box
    
    return image_data, box_data
