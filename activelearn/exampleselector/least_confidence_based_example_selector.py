import operator
import numpy as np

from activelearn.exampleselector.uncertainity_based_example_selector import UncertainityBasedExampleSelector
from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr

class LeastConfidenceExampleSelector(UncertainityBasedExampleSelector):
    
    def __init__(self):
        super(LeastConfidenceExampleSelector, self).__init__()

    
    def _compute_margin(self, probability):
        return (probability[0] - probability[1])
    
    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, batch_size=1):
            """
            Used to select informative examples based on the confidence of the examples.
    
            Args:
                model (Model): Model that is used to compute the uncertainty measure of the example
                unlabeled_dataset (int): 
                exclude_attr (string): 
                batch_size (boolean): 
        
            Attributes:
                model (Model): 
                example_selector (int): An attribute to store the overlap threshold value.
                labeler (string): An attribute to store the comparison operator.
                batch_size (boolean): An attribute to store the value of the flag 
                    allow_missing.
                
            """
            validate_input_table(unlabeled_dataset, 'unlabeled dataset')
            #validate exclude attr
            for attr in exclude_attrs:
                validate_attr(attr, unlabeled_dataset.columns, "attr", 'unlabeled_dataset')
                 
            feature_attrs = remove_exclude_attr(list(unlabeled_dataset.columns), exclude_attrs, unlabeled_dataset)

            feature_values = unlabeled_dataset[feature_attrs]
            
            probabilities = model.predict_proba(feature_values)

            # compute the maximum probability of being classified into any class for the unlabeled pairs
            maxprobabilities = np.max(probabilities, axis=1)
            next_batch_idxs = np.argpartition(maxprobabilities, -batch_size)[:batch_size]
            return unlabeled_dataset.iloc[next_batch_idxs]
    