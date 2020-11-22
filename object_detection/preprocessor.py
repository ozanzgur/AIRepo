to_import = [
    'utils',
    'object_detection.yolov3.yoloutils'
    ]

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import logging
logger = logging.getLogger('pipeline')

import numpy as np
from PIL import Image, ImageFont, ImageDraw

class Preprocessor:
    @utils.catch('YOLOPREPROCESSOR_INITERROR')
    def __init__(self, **kwargs):
        def_args = dict(
            image_size = (416, 416),
        )
        # Use arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})

        self.model_image_size = self.image_size
    
    @utils.catch('YOLOPREPROCESSOR_CALLERROR')
    def __call__(self, x, debug = False):
        for x_example in x:
            x_example.update({
                'image_orig': x_example['image'].copy(),
                'image': self.transform(x_example['image'])
            })
        return x
    
    # Preprocessing for prod takes place in yolo.py for yolo
    @utils.catch('YOLOPREPROCESSOR_TRANSFORMERROR')
    def transform(self, image, debug = False):        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = yoloutils.letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = yoloutils.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        logger.info(f'Preprocessed image size: {image_data.shape}')
        return image_data
