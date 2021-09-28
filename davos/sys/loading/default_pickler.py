import pickle
# noinspection PyUnresolvedReferences
from pickle import *


class Unpickler(pickle.Unpickler):
    """ Unpickler that knows about modules that have been moved around or renamed. """

    def find_class(self, module: str, name):
        try:
            return super().find_class(module, name)
        except:
            if module == 'ltr.admin.local' and name == 'EnvironmentSettings':
                return super().find_class('davos.config', 'EnvSettings')
            s = 'ltr.admin.'
            if module.startswith(s):
                return super().find_class('davos.sys.' + module[len(s):], name)
            if module == 'ltr.models.target_classifier.features':
                return super().find_class('davos.models.blocks', name)
            if module == 'ltr.models.layers.normalization':
                return super().find_class('davos.models.blocks', name)
            if module == 'ltr.models.meta.steepestdescent':
                return super().find_class('davos.models.steepestdescent', name)
            s = "ltr."
            if module.startswith(s):
                if name == 'LWTLNet':
                    name = 'LWLNet'
                elif name == 'LWTLResidual':
                    name = 'LWLResidual'
                return super().find_class('davos.' + module[len(s):], name)
            if module == "pytracking.libs.tensordict":
                return super().find_class('davos.lib.tensordict', name)
            raise
