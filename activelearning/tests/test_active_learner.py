import unittest

from nose.tools import *
import pandas as pd
import numpy as np
import operator
import os

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from activelearning.exampleselector.entropy_based_example_selector import EntropyBasedExampleSelector
from activelearning.labeler.cli_labeler import CliLabeler
from activelearning.activelearner.active_learner import ActiveLearner

class ActiveLearnerTests(unittest.TestCase):
    
    def sample_get_instruction_fn(self, context):
        return "Can you enter the label y or n?"
    def get_example_display_fn(self, example, context):
        return example

    def setUp(self):

#         dataset_a=pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/table_A.csv')).head(1000)
#         dataset_b=pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/table_B.csv')).head(1000)
        # labeled data, typically small in number in DataFrame format
        labeled_dataset_seed = pd.read_csv(os.path.join(os.path.dirname(__file__), '/Data/seed.csv'),  sep='\t')
        
        self.unlabeled_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '/Data/seed.csv'), sep='\t')
        self.model = linear_model.LogisticRegression()
        
        feature_attrs = list(self.unlabeled_dataset.columns)
        feature_attrs.remove('_id')
        feature_attrs.remove('l_ID')
        feature_attrs.remove('r_ID')
        self.model.fit(self.unlabeled_dataset[feature_attrs].values[:3],
                                            labeled_dataset_seed['label'].values[:3])
        
        #create a model
        model = RandomForestClassifier()   
        #create a labeler
        #Initialize the EMCliLabeler
        labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
    
        #create a selector
        selector  = EntropyBasedExampleSelector()
        #create a learner
        alearner = ActiveLearner(model, selector, labeler, 1, 1)
        
        alearner.learn(self.unlabeled_dataset, labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=None, label_attr='label')

    #testing non batch mode
    def test_learn(self):
        assert_equal(0,0)
    
        