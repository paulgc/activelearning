 
import operator
import numpy as np
 
from activelearn.exampleselector.uncertainity_based_example_selector import UncertainityBasedExampleSelector
from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr

class SmallestMarginBasedExampleSelector(UncertainityBasedExampleSelector):
     
    def __init__(self):
        super(SmallestMarginBasedExampleSelector, self).__init__()
     
    def _compute_margin(self, probability):
        return (probability[0] - probability[1])
     
    def _utility_example(self, probability):
        return self._compute_margin(probability)
     
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, batch_size=1):
        """
        Used to select informative examples based on the margin.

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
        
        # remove exclude attrs                                                  
        
        feature_attrs = remove_exclude_attr(list(unlabeled_dataset.columns), exclude_attrs, unlabeled_dataset)
        
        feature_values = unlabeled_dataset[feature_attrs]
            
        probabilities = model.predict_proba(feature_values)

        # compute the entropy for the unlabeled pairs
        margins = {}
        for i in xrange(len(probabilities)):
            margins[i] = self._compute_margin(probabilities[i])
        # compute the margin of uncertainity for the unlabeled pairs
        candidate_examples = sorted(margins.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(margins))]
        next_batch_idxs = [val[0] for val in candidate_examples]
        return unlabeled_dataset.iloc[next_batch_idxs]
     
