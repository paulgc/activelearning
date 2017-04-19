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
from activelearning.utils.validation import validate_attr
class ActiveLearnerTests(unittest.TestCase):
    
    def sample_get_instruction_fn(self, context):
        return "Can you enter the label y or n?"

    def get_example_display_fn(self, example, context):
        table_a = context["dataset_a"]
        table_b = context["dataset_b"]
        example_A = table_a[table_a["A.ID"] == example["l_ID"]]
        example_B = table_b[table_b["B.ID"] == example["r_ID"]]
        return str(example_A) + "\n" + str(example_B)

    def setUp(self):
        dataset_a=pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/table_A.csv')).head(1000)
        dataset_b=pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/table_B.csv')).head(1000)
        self.context = {"dataset_a": dataset_a, "dataset_b": dataset_b }
        # labeled data, typically small in number in dataFrame format
        self.labeled_dataset_seed = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/seed.csv'),  sep='\t')
    
        self.unlabeled_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/sample_fvs.csv'), sep='\t')
        
        #create a model
        self.model = RandomForestClassifier()   
        #create a labeler
        #Initialize the EMCliLabeler
        self.labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
    
        #create a selector
        self.selector  = EntropyBasedExampleSelector()
        

    #testing non batch mode
    def test_active_learn_non_batch(self):
        #create a learner
        alearner = ActiveLearner(self.model, self.selector, self.labeler, 1, 2)
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')
        assert_equal(0,0)
    
    #testing batch mode
    def test_active_learn_batch(self):
        #create a learner
        alearner = ActiveLearner(self.model, self.selector, self.labeler, 2, 2)
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')
        assert_equal(0,0)    
    
    def test_active_learn_different_model(self):
        #create a learner
        alearner = ActiveLearner(self.model, self.selector, self.labeler, 2, 2)
        alearner.learn(self.unlabeled_dataset, self.labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=self.context, label_attr='label')
        assert_equal(0,0)    
         