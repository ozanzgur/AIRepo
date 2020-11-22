"""

I don't use this anymore.

"""

to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import importlib
import time
import logging
logger = logging.getLogger('pipeline')

# Experiment logging
try:
    from mlflow import log_metric, log_param, log_artifact, start_run, end_run
    mlflow_available = True
except:
    logger.warning('MLFLOW NOT IMPORTED.')
    mlflow_available = False


class Pipeline:
    @utils.catch('PIPELINE_INITERROR')
    def __init__(self, definitions_path, param_json_path = None, **kwargs):
        # Get parameters
        self.parameters = importlib.import_module(name = definitions_path)
        if not param_json_path is None:
            print(f'Loading parameters from {param_json_path}')
            self.parameters.import_from_json(param_json_path)
        
        self.__dict__.update(self.parameters.PipelineParameters.__dict__)
        self.__dict__.update(kwargs)
        
        self.flow = []
        self.step_names = []
        self.mlflow_logging = mlflow_available
        
        utils.set_logger(self.log_level)
        logger = logging.getLogger('pipeline')
        
        self.init_flow()
    
    @utils.catch('PIPELINE_INITFLOWERROR')
    def init_flow(self):
        for step_dict in self.workflows[self.workflow]:
            class_name = step_dict['class_name']
            module_path = step_dict['module_path']
            step_name = step_dict.get('step_name', class_name)

            # equivalent of:
            # from package import Module
            # self.module = Module(**all_params)
            new_module = importlib.import_module(name = module_path)
            new_class = getattr(new_module, class_name)(
                step_name = step_name,
                class_name = class_name,
                data_dir = self.data_dir,
                preprocessor_method = self.preprocessor_method,
                model_method = self.model_method,
                **step_dict.get('params', {})
                )
            self.__dict__.update({step_name: new_class})
            
            self.flow.append(new_class)
            self.step_names.append(step_name)
    
    @utils.catch('PIPELINE_PREDICTERROR')
    def predict(*args, **kwargs):
        self.run(*args, **kwargs)
    
    @utils.catch('PIPELINE_RUNERROR')
    def run(self, x = None, DocId = '', **kwargs):

        if self.workflow == 'train':
            start_run(
                run_name = self.project_name
                )

        start_time = time.time() if self.debug else None
        outputs = [[{'output': x, 'doc_id': DocId}]]
        
        for step, step_name in zip(self.flow, self.step_names):
            logger.info(f'Running {step_name}...')
            outputs.append(step(outputs[-1], debug = self.debug, **kwargs))
            if not (isinstance(outputs[-1], list) and all([isinstance(i, dict) for i in outputs[-1]])):
                logger.warning(f'*** WARNING: output of {step_name} is not a list of dictionaries. ***')

            # Timing
            if self.debug:
                curr_time = time.time()
                logger.info(f'{step_name} time: {curr_time - start_time}')
                start_time = curr_time 

        if self.mlflow_logging and self.workflow in ['train', 'test']:
            params_path = self.parameters.export_to_json()
            log_artifact(params_path)

        if self.workflow == 'train':
            end_run()


        if self.debug:
            return {step_name: y for step_name, y in zip(self.step_names, outputs[1:])}
        else:
            return outputs[-1]