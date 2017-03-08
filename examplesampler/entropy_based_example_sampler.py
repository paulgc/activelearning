'''
Created on Mar 4, 2017

@author: lokananda
'''
import operator
import numpy as np

from uncertainity_based_example_sampler import UncertainityBasedExampleSampler
from util.weighted_random_sampler import  WeightedRandomSampler

class EntropyBasedExampleSampler(UncertainityBasedExampleSampler):
    
    def __init__(self, model, batch_mode, batch_size=5):
        super(EntropyBasedExampleSampler, self).__init__(model)
        self.batch_mode = batch_mode
        self.batch_size = batch_size
    
    def _compute_entropy(self, probability):
        return np.sum(-probability * np.log(probability))
    
    def utilityExample(self, probability):
        return self._compute_entropy(probability)
    
    
    def next_examples(self, unlabeled_dataset, feature_attrs):
        # compute the prediction probabilities for the unlabeled dataset
        if self.batch_mode:
            probabilities = self.model.predict_proba(unlabeled_dataset[feature_attrs].values) 
            # compute the entropy for the unlabeled pairs
            entropies = {}
            for i in xrange(len(probabilities)):
                entropies[i] = self._compute_entropy(probabilities[i])
    
            candidate_examples = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)[:min(self.batch_size, len(entropies))]
            next_batch_idxs = []
            next_batch_idxs = map(lambda val: val[0], candidate_examples)
            return unlabeled_dataset.iloc[next_batch_idxs]
        else:
            
            probabilities = self.model.predict_proba(unlabeled_dataset[feature_attrs].values)
            entropy = np.sum(-probabilities * np.log(probabilities), axis=1)
            example_id = np.argmax(entropy)
            return unlabeled_dataset.iloc[example_id]