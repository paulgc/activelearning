'''
Created on Mar 4, 2017

@author: lokananda
'''
import unittest

from nose.tools import *
import pandas as pd
import numpy as np
import operator
import os

from labeler.em_cli_labeler import EntityMatchingCliLabeler


class EntropySamplerTests(unittest.TestCase):
    def setUp(self):
        self.feature_vs = pd.read_csv("Data/sample_data.csv", sep='\t')
        self.table_A = pd.read_csv("Data/table_A.csv", sep=',')
        self.table_B = pd.read_csv("Data/table_B.csv", sep=',')
        

    def test_label(self):
        eml = EntityMatchingCliLabeler(self.table_A, self.table_B, 'A.ID', 'B.ID', 'l_ID', 'r_ID', ['birth_year','hourly_wage'], ['birth_year','hourly_wage'])
        self.feature_vs['label'] = pd.Series(np.random.randn(self.feature_vs.size))
        eml.label(self.feature_vs, label_attr='label')