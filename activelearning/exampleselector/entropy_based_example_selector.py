
import operator
import numpy as np
import activelearning.utils.validation
from activelearning.exampleselector.uncertainity_based_example_selector import UncertainityBasedExampleSelector

from activelearning.utils.validation import validate_input_table
from activelearning.utils.validation import validate_attr

class EntropyBasedExampleSelector(UncertainityBasedExampleSelector):
    
    def __init__(self):
        super(EntropyBasedExampleSelector, self).__init__()
    
    def _compute_entropy(self, probability):
        return np.sum(-probability * np.log(probability))
    
    def select_examples(self, unlabeled_dataset, model, exclude_attrs=None, 
                        batch_size=1):
        
        # check if the input candset is a dataframe
        validate_input_table(unlabeled_dataset, 'unlabeled dataset')
        
        #validate exclude attr
        for attr in exclude_attrs:
            validate_attr(attr, unlabeled_dataset.columns, "attr", 'unlabeled_dataset')
        
        # remove exclude attrs
        feature_attrs = list(unlabeled_dataset.columns)
        if exclude_attrs:
            for attr in exclude_attrs:
                feature_attrs.remove(attr)

        # compute the prediction probabilities for the unlabeled dataset
        probabilities = model.predict_proba(unlabeled_dataset[feature_attrs].values) 

        # compute the entropy for the unlabeled pairs
        entropies = {}
        for i in xrange(len(probabilities)):
            entropies[i] = self._compute_entropy(probabilities[i])

        candidate_examples = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(entropies))]
        next_batch_idxs = map(lambda val: val[0], candidate_examples)
        return unlabeled_dataset.iloc[next_batch_idxs]
