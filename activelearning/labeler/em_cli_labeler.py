
import itertools


from labeler.cli_labeler import CliLabeler

class EntityMatchingCliLabeler(CliLabeler):
    
    def __init__(self, table_A, table_B, A_id_attr, B_id_attr, fvs_A_id_attr, fvs_B_id_attr, A_out_attrs, B_out_attrs):
        super(EntityMatchingCliLabeler, self).__init__(self)
        self.table_A = table_A
        self.table_B = table_B
        self.A_id_attr = A_id_attr
        self.B_id_attr = B_id_attr
        self.fvs_A_id_attr = fvs_A_id_attr
        self.fvs_B_id_attr = fvs_B_id_attr
        self.A_out_attrs = A_out_attrs
        self.B_out_attrs = B_out_attrs
        
    def _format_tuple_pair_for_labelling(self, candidate_pair):
        '''
            Display only the attributes specified by user for labeling
        '''
        banner_str = "Select whether the given below pair is a Match(1) or Non Match(0)" + "\n"
        return banner_str + str(candidate_pair[0][self.A_out_attrs]) + "\n" + str(candidate_pair[1][self.B_out_attrs]) + "\n"
        
    def label(self, examples_to_label, label_attr='label'):
        
        table_A_ids = examples_to_label[self.fvs_A_id_attr]
        table_B_ids = examples_to_label[self.fvs_B_id_attr]
        
        #list of original examples to label
        pairs_to_label = []
         
        for table_A_id,table_B_id in itertools.izip(table_A_ids,table_B_ids):
            table_A_tuple = self.table_A.where(self.table_A[self.A_id_attr] == table_A_id).dropna().head(1) 
            table_B_tuple = self.table_B.where(self.table_B[self.B_id_attr] == table_B_id).dropna().head(1)
            pairs_to_label.append((table_A_tuple, table_B_tuple))

        
        for c_pair in pairs_to_label:
            x = self._input_from_stdin(self._format_tuple_pair_for_labelling(c_pair))
            examples_to_label.loc[((examples_to_label[self.fvs_A_id_attr] == c_pair[0].iloc[0][self.A_id_attr]) & (examples_to_label[self.fvs_B_id_attr] == c_pair[1].iloc[0][self.B_id_attr])), label_attr] = x
        return examples_to_label
    
    
    
    
