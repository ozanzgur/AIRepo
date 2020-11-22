to_import = [
    'utils'
    ]

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import logging
logger = logging.getLogger('pipeline')

class Postprocessor:
    @utils.catch('POSTPROC_INITERROR')
    def __init__(self, **kwargs):
        pass
    
    @utils.catch('POSTPROC_PROCESSERROR')
    def __call__(self, x, debug = False):
        assert(len(x) == 1)

        max_confidence = 0
        best_confidences = None
        best_class = -1
        
        out_boxes, out_scores, out_classes = x[0]['output']
        n_boxes = len(out_boxes)
        logger.info(f'n_boxes: {n_boxes}')

        for i in range(n_boxes):
            logger.info(f'box: {out_boxes[i]}, score: {out_scores[i]}, class: {out_classes[i]}')
        
        x[0].update({
            'output': [
                {
                    'class': c,
                    'confidence': conf,
                    'box': box
                } for c, conf, box in zip(out_classes, out_scores, out_boxes)]
            })
        
        return x[0]