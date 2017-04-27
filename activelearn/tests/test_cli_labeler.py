
import unittest

from nose.tools import *
import pandas as pd
import numpy as np
import operator
import os

from activelearning.labeler.cli_labeler import CliLabeler


class CliLabelerTests(unittest.TestCase):
    def setUp(self):

        self.table_A = pd.read_csv(os.path.join(os.path.dirname(__file__) + "/data/table_A.csv"), sep=',')
        self.table_B = pd.read_csv(os.path.join(os.path.dirname(__file__) + "/data/table_B.csv"), sep=',')
        
        self.fvs_A_id_attr = 'l_ID'
        self.fvs_B_id_attr = 'r_ID'
        self.A_id_attr = 'A.ID'
        self.B_id_attr = 'B.ID'
        
        self.A_out_attrs = ['A.ID', 'birth_year','hourly_wage']
        self.B_out_attrs = ['B.ID', 'birth_year','hourly_wage']


    def default_get_instruction_fn(self):
        banner_str = "Select whether the given below pair is a Match(y) or Non Match(n)" + "\n"
        return banner_str

    def display_tuple_pair_for_label(self, examples_to_label):
        
        #obtaining the raw representation
        table_A_id = examples_to_label[self.fvs_A_id_attr]
        table_B_id = examples_to_label[self.fvs_B_id_attr]
        
        raw_tuple_table_A = self.table_A.where(self.table_A[self.A_id_attr] == table_A_id).dropna().head(1)
        raw_tuple_table_B = self.table_B.where(self.table_B[self.B_id_attr] == table_B_id).dropna().head(1)
        
        #'A.ID', 'B.ID', 'l_ID', 'r_ID', ['birth_year','hourly_wage'], ['birth_year','hourly_wage']
        return str(raw_tuple_table_A[self.A_out_attrs]) + "\n" + str(raw_tuple_table_B[self.B_out_attrs]) + "\n" + self.prompt_msg
        
    @raises(TypeError)
    def test_cli_labeler_invalid_unlabeled_dataset(self):
        labeler = CliLabeler()
        context = {"display_fn": self.display_tuple_pair_for_label, "display_instruction_fn":self.default_get_instruction_fn}
        label_attr = 'label'
        labeler.label([], context, label_attr)

#     def test_label(self):
#         eml = CliLabeler(self.default_get_instruction_fn, self.display_tuple_pair_for_label, labels= {"y":0, "n":1}, label_attr='label')
#         eml.label(self.feature_vs)
