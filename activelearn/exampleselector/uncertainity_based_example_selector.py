
from activelearn.exampleselector.example_selector import ExampleSelector

class UncertainityBasedExampleSelector(ExampleSelector):
    """
    Base class for all uncertainty based example selectors
    """
    def __init__(self):
        super(UncertainityBasedExampleSelector, self).__init__()
        
    def _utility_example(self):
        raise NotImplementedError( "Should have implemented this" )
