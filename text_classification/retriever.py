to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################
    
from PIL import Image
import base64
import json
import io

import logging
logger = logging.getLogger('pipeline')

class Retriever:
    @utils.catch('RETRIEVER_INITERROR')
    def __init__(self, **kwargs):
        def_args = dict()

        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
    
    @utils.catch('RETRIEVER_PROCESSERROR')
    def __call__(self, x, debug = False):
        """Takes a json message with field 'texts', returns
        a list of dicts with field 'output'.
        """
        x = {'output': [x]}
        return x