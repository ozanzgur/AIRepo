from os.path import dirname, abspath, basename
import sys
import importlib

def import_modules(name, file, to_import, level = 1):
    project_path = abspath(__file__)
    for _ in range(level):
        project_path = dirname(project_path)

    project_name = basename(project_path)

    # Insert package path to path
    project_package_path = dirname(project_path)
    if not project_package_path in sys.path:
        sys.path.insert(0, project_package_path)

    # Import modules
    for module in to_import:
        module_name = module.split('.')[-1]
        new_module = importlib.import_module(name = f'.{module}', package = project_name)
        sys.modules[name].__dict__.update({module_name: new_module})