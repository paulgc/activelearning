import operator
import numpy as np

from activelearning.exampleselector.uncertainity_based_example_selector import UncertainityBasedExampleSelector
from activelearning.utils.validation import validate_input_table
from activelearning.utils.validation import validate_attr

class LeastConfidenceMeasureExampleSelector(UncertainityBasedExampleSelector):
    
    def __init__(self, model, batch_mode, batch_size=5):
        super(LeastConfidenceMeasureExampleSelector, self).__init__(model)

    
    def _compute_margin(self, probability):
        return (probability[0] - probability[1])
    
    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, batch_size=1):
            
            validate_input_table(unlabeled_dataset, 'unlabeled dataset')
            
            #validate exclude attr
            for attr in exclude_attrs:
                validate_attr(attr, unlabeled_dataset.columns, "attr", 'unlabeled_dataset')
            
            
            # remove exclude attrs
            feature_attrs = list(unlabeled_dataset.columns)
            if exclude_attrs:
                for attr in exclude_attrs:
                    feature_attrs.remove(attr)
            
            feature_values = unlabeled_dataset[feature_attrs]
            
            try:
                probabilities = model.predict_proba(feature_values) 
            except Exception as e:
                print e

            # compute the maximum probability of being classified into any class for the unlabeled pairs
            maxprobabilities = np.max(probabilities, axis=1)
            next_batch_idxs = np.argpartition(maxprobabilities, -batch_size)[:batch_size]
            return unlabeled_dataset.iloc[next_batch_idxs]
    