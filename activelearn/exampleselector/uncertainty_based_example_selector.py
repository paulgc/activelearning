
from activelearn.exampleselector.example_selector import ExampleSelector
from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr

class UncertaintyBasedExampleSelector(ExampleSelector):
    """
    Base class for all uncertainty based example selectors
    """
    def __init__(self):
        super(UncertaintyBasedExampleSelector, self).__init__()
        
    def _utility_example(self):
        raise NotImplementedError( "Should have implemented this" )

    def _validate_tables_and_calc_probabilities(self, unlabeled_dataset, model, exclude_attrs=None):
                # check if the input candset is a dataframe
        validate_input_table(unlabeled_dataset, 'unlabeled dataset')
        
        #validate exclude attr
        for attr in exclude_attrs:
            validate_attr(attr, unlabeled_dataset.columns, "attr", 'unlabeled_dataset')
        
        # remove exclude attrs
        feature_attrs = remove_exclude_attr(list(unlabeled_dataset.columns), exclude_attrs, unlabeled_dataset)

        feature_values = unlabeled_dataset[feature_attrs]
        
        # compute the prediction probabilities for the unlabeled dataset
        probabilities = model.predict_proba(feature_values)
        
        return probabilities
    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, 
                        batch_size=1):
        probabilities = self._validate_tables_and_calc_probabilities(unlabeled_dataset, model, exclude_attrs)
        return self.select_uncertain_examples(probabilities, unlabeled_dataset, batch_size)
        