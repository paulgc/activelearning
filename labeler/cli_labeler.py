'''
Created on Mar 7, 2017

@author: lokananda
'''
from labeler.labeler_base import LabelerBase

class CliLabeler(LabelerBase):
    
    def _input_from_stdin(self, banner_str):
        return input(banner_str)
    
    def label(self, feature):
        x = self._input_from_stdin()
        return x
    