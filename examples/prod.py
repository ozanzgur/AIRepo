to_import = ['utils',
             'config_loader',
             'object_detection.preprocessor',
             'object_detection.yolo',
             'object_detection.postprocessor']

import importer
importer.import_modules(__name__, __file__, to_import)

import logging
import os
logger = logging.getLogger('pipeline')
Config = config_loader.ConfigLoader(
    'config.json'
    )

################################################################################

Preprocessor = preprocessor.Preprocessor(**Config["preprocessor_params"])
Yolo = yolo.YOLOV3(**Config["yolo_params"])
Postprocessor = postprocessor.Postprocessor()

#################################################################################

@utils.catch('PIPELINE_PREDICTERROR')
def predict(image_id = '', image  = None, **kwargs):
    assert(image is not None)

    x = [{'image': image}]
    x = Preprocessor(x)
    x = Yolo(x)
    x = Postprocessor(x)

    return x
    
