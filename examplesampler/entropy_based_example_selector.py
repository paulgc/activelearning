'''
Created on Mar 4, 2017

@author: lokananda
'''
import operator
import numpy as np

from examplesampler.uncertainity_based_example_selector import UncertainityBasedExampleSelector
from util.weighted_random_sampler import  WeightedRandomSampler

class EntropyBasedExampleSelector(UncertainityBasedExampleSelector):
    
    def __init__(self, model):
        super(EntropyBasedExampleSelector, self).__init__(model)
    
    def _compute_entropy(self, probability):
        return np.sum(-probability * np.log(probability))
    
    def select_examples(self, unlabeled_dataset, feature_attrs, batch_size=1):
        # compute the prediction probabilities for the unlabeled dataset
        probabilities = self.model.predict_proba(unlabeled_dataset[feature_attrs].values) 
        # compute the entropy for the unlabeled pairs
        entropies = {}
        for i in xrange(len(probabilities)):
            entropies[i] = self._compute_entropy(probabilities[i])

        candidate_examples = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(entropies))]
        next_batch_idxs = []
        next_batch_idxs = map(lambda val: val[0], candidate_examples)
        return unlabeled_dataset.iloc[next_batch_idxs]
