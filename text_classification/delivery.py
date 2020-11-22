to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import numpy as np
import json

import logging
logger = logging.getLogger('pipeline')

class Delivery:
    @utils.catch('DELIVERY_INITERROR')
    def __init__(self, **kwargs):
        def_args = dict()
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
    
    @utils.catch('DELIVERY_PROCESSERROR')
    def __call__(self, x, debug = False):        
        conf = x['output'].flatten()
        final_result = {
            'Results': {'SelectedResult': str(int(np.argmax(conf))),
                        'AllResults': [{'Value': str(i), 'Confidence': str(conf[i])} for i in range(len(conf))]
            },
            'ImageId': str(x['doc_id']),
            'ErrorMessage': ''
        }
        
        """output_message = json.dumps(final_result,indent=4)
        return output_message"""
        return final_result