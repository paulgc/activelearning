
from activelearning.exampleselector.example_selector import ExampleSelector

class UncertainityBasedExampleSelector(ExampleSelector):
    """
    should support querying batch samples
    """
    def __init__(self):
        super(UncertainityBasedExampleSelector, self).__init__()
        
    def _utility_example(self):
        raise NotImplementedError( "Should have implemented this" )
