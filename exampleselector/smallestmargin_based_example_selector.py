
import operator
import numpy as np

from examplesampler.uncertainity_based_example_selector import UncertainityBasedExampleSelector

class SmallestMarginBasedExampleSelector(UncertainityBasedExampleSelector):
    
    def __init__(self, batch_mode, batch_size=5):
        super(SmallestMarginBasedExampleSelector, self).__init__()
        self.batch_mode = batch_mode
        self.batch_size = batch_size
    
    def _compute_margin(self, probability):
        return (probability[0] - probability[1])
    
    def _utility_example(self, probability):
        return self._compute_margin(probability)
    
    def next_examples(self, unlabeled_dataset, model, exclude_attrs=None):
        # remove exclude attrs                                                  
        feature_attrs = list(unlabeled_dataset.columns)                         
        if exclude_attrs:                                                       
            for attr in exclude_attrs:                                          
                feature_attrs.remove(attr)   

        if self.batch_mode:
            probabilities = model.predict_proba(unlabeled_dataset[feature_attrs].values) 
            # compute the entropy for the unlabeled pairs
            margins = {}
            for i in xrange(len(probabilities)):
                margins[i] = self._compute_margin(probabilities[i])
            # compute the margin of uncertainity for the unlabeled pairs
            candidate_examples = sorted(margins.items(), key=operator.itemgetter(1), reverse=True)[:min(self.batch_size, len(margins))]
            next_batch_idxs = []
            next_batch_idxs = map(lambda val: val[0], candidate_examples)
            return unlabeled_dataset.iloc[next_batch_idxs]
        else:
            probabilities = model.predict_proba(unlabeled_dataset[feature_attrs].values)
            entropy = np.diff(probabilities, axis=1)
            example_id = np.argmax(entropy)
            return unlabeled_dataset.iloc[example_id]
    
