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
        table_a = context["dataset_a"]
        table_b = context["dataset_b"]
        example_A = table_a[table_a["A.ID"] == example["l_ID"]]
        example_B = table_b[table_b["B.ID"] == example["r_ID"]]
        return example_A + example_B

    def setUp(self):
        dataset_a=pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/table_A.csv')).head(1000)
        dataset_b=pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/table_B.csv')).head(1000)
        context = {"dataset_a": dataset_a, "dataset_b": dataset_b }
        # labeled data, typically small in number in DataFrame format
        labeled_dataset_seed = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/seed.csv'),  sep='\t')
    
        self.unlabeled_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/seed.csv'), sep='\t')

        #create a model
        model = RandomForestClassifier()   
        #create a labeler
        #Initialize the EMCliLabeler
        labeler = CliLabeler(self.sample_get_instruction_fn, self.get_example_display_fn, {'y': 1, 'n': 0})
    
        #create a selector
        selector  = EntropyBasedExampleSelector()
        #create a learner
        alearner = ActiveLearner(model, selector, labeler, 1, 1)
        alearner.learn(self.unlabeled_dataset, labeled_dataset_seed, exclude_attrs=['_id', 'l_ID', 'r_ID'], context=context, label_attr='label')

    #testing non batch mode
    def test_learn(self):
        assert_equal(0,0)
        
        