
import unittest

from nose.tools import *
import pandas as pd
import numpy as np
import operator
import os

from sklearn import linear_model
from exampleselector.entropy_based_example_selector import EntropyBasedExampleSelector

class EntropySelectorTests(unittest.TestCase):
    def setUp(self):

#         dataset_a=pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/table_A.csv')).head(1000)
#         dataset_b=pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/table_B.csv')).head(1000)
        # labeled data, typically small in number in DataFrame format
        labeled_dataset_seed = pd.read_csv('/Users/lokananda/Documents/IndependentStudy/activelearning/tests/productsSeedMod.csv')
        
        # merge the original unlabeled data with labeled Data
        #self.unlabeled_dataset = self.getPartiallyLabelledDatset(dataset_a, dataset_b, labeled_dataset_seed)
        
        self.unlabeled_dataset = pd.read_csv("Data/sample_data.csv", sep='\t')
        self.model = linear_model.LogisticRegression()
        
        feature_attrs = list(self.unlabeled_dataset.columns)
        feature_attrs.remove('_id')
        feature_attrs.remove('l_ID')
        feature_attrs.remove('r_ID')
        self.model.fit(self.unlabeled_dataset[feature_attrs].values[:3],
                                            labeled_dataset_seed['label'].values[:3])
        
        

    #testing non batch mode
    def test_next_example(self):
        es = EntropyBasedExampleSelector()
        
        instance_to_be_labeled = es.select_examples(
                                     self.unlabeled_dataset.head(5), self.model, 
                                     ['_id', 'l_ID', 'r_ID'])
  
        assert_equal(0,instance_to_be_labeled["_id"])
    
    
    #testing batch mode
    def test_select_examples(self):
        es = EntropyBasedExampleSelector()
        
        instances_to_be_labeled = es.select_examples(
                                      self.unlabeled_dataset.head(5), self.model,
                                      ['_id', 'l_ID', 'r_ID'])

        assert_equal(0,instances_to_be_labeled.iloc[0]["_id"])
        assert_equal(3,instances_to_be_labeled.iloc[1]["_id"])
        
        
    def getPartiallyLabeledDatset(self, ldf, rdf, labeled_dataset_seed):
        #for now read the feature vectors of sample pairs directly from file
        fvs = pd.read_csv("Data/sample_data.csv", sep='\t')
        
        #manually label a few datapoints
        y_train = [1]*(len(fvs)/2)
        y_train += [0]*(len(fvs)/2)
        n_labeled = len(y_train)
        n_unlabeled = len(fvs) - len(y_train)
        y_train += [None]*n_unlabeled
        
    
        return fvs

if __name__ == '__main__':
    unittest.main()
