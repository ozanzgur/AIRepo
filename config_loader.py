from os.path import dirname, abspath, basename, join
import logging
import json
import re

logger = logging.getLogger('pipeline')

#project_path = dirname(abspath(__file__))

class ConfigLoader:
    def __init__(self, config_path = 'config.json', project_path = '', **kwargs):
        #self.config_name = config_name
        self.config_path = config_path#join(project_path, config_name)
        logger.info(f'Loading config from: {self.config_path}')
        
        self.data_dir = join(project_path, 'data')
        self.params = from_json(self.config_path)
        self.project_path = project_path
        self.__dict__.update(kwargs)
    
    def replace_var(self, text):
        """
        Try to obtain parameter from this class. If it fails,
        try to get it from config json
        
        """
        m = re.search('{(.+?)}', text)
        if m:
            var_name = m.group(1)
            try:
                var_ref = getattr(self, var_name)
            except:
                var_ref = self.params[var_name]
            
            # If variable is a string, place it "{var1}_afsdv" => "abcd_afsdf"
            # Otherwise, return variable
            if isinstance(var_ref, str):
                to_replace = '{' + var_name + '}'
                return text.replace(to_replace, var_ref)
            else:
                return var_ref
            
        else:
            return text
    
    def __getitem__(self, arg):
        param_set = self.params[arg]
        formatted_param_set = None
        
        if isinstance(param_set, dict):
            formatted_param_set = param_set.copy()
            
            for k,v in param_set.items():
                if isinstance(v, str):
                    formatted_param_set[k] = self.replace_var(param_set[k])
                    if formatted_param_set[k] == 'None':
                        formatted_param_set[k] = None
                elif isinstance(v, dict):
                    for k_l2, v_l2 in v.items():
                        v[k_l2] = self.replace_var(v[k_l2])
                        if v[k_l2] == 'None':
                            v[k_l2] = None
                    
            # Add data dir to parameters of all steps
            formatted_param_set['data_dir'] = self.data_dir
        else:
            formatted_param_set = param_set
                    
        return formatted_param_set

def from_json(json_path):
    logger.info(f'Loading config from {json_path}')
    with open(json_path, "r") as f:
        params = json.load(f)
    return params