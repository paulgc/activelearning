'''
Created on Mar 4, 2017

@author: lokananda
'''
from examplesampler.example_selector import ExampleSelector
class UncertainityBasedExampleSelector(ExampleSelector):
    """
    should support querying batch samples
    """
    def __init__(self, model):
        super(UncertainityBasedExampleSelector, self).__init__()
        self.model = model
        
    def _utility_example(self):
        raise NotImplementedError( "Should have implemented this" )