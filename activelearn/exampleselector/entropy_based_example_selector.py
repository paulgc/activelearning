
import operator
import pandas as pd
import time
from activelearn.exampleselector.uncertainity_based_example_selector import UncertainityBasedExampleSelector

from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr

class EntropyBasedExampleSelector(UncertainityBasedExampleSelector):
    """
    Entropy based Uncertainty example selection
    """
    def __init__(self):
        super(EntropyBasedExampleSelector, self).__init__()
    
    
    def _compute_entropy(self, probability):
        if 0 in probability:
            return 0
        else:
            return pd.np.sum(-probability * pd.np.log(probability))
    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, 
                        batch_size=1):
        """
        Used to select informative examples based on the entropy of the examples.

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

        # compute the entropy for the unlabeled pairs
        entropies = pd.np.sum(-probabilities * pd.np.log(probabilities), axis=1)
        
        entropies = dict(enumerate(entropies))
        
        candidate_examples = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(entropies))]

        next_batch_idxs = map(lambda val: val[0], candidate_examples)
        return unlabeled_dataset.iloc[next_batch_idxs]
