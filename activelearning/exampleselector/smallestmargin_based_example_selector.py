
import operator
import numpy as np

from activelearning.exampleselector.uncertainity_based_example_selector import UncertainityBasedExampleSelector

class SmallestMarginBasedExampleSelector(UncertainityBasedExampleSelector):
    
    def __init__(self):
        super(SmallestMarginBasedExampleSelector, self).__init__()
    
    def _compute_margin(self, probability):
        return (probability[0] - probability[1])
    
    def _utility_example(self, probability):
        return self._compute_margin(probability)
    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, batch_size=1):
        # remove exclude attrs                                                  
        feature_attrs = list(unlabeled_dataset)                         
        if exclude_attrs:                                                       
            for attr in exclude_attrs:                                          
                feature_attrs.remove(attr)   

            probabilities = model.predict_proba(unlabeled_dataset[feature_attrs].values) 
            # compute the entropy for the unlabeled pairs
            margins = {}
            for i in xrange(len(probabilities)):
                margins[i] = self._compute_margin(probabilities[i])
            # compute the margin of uncertainity for the unlabeled pairs
            candidate_examples = sorted(margins.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(margins))]
            next_batch_idxs = []
            next_batch_idxs = map(lambda val: val[0], candidate_examples)
            return unlabeled_dataset.iloc[next_batch_idxs]
    
