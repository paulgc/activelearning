
import operator
import pandas as pd
import time
from activelearning.exampleselector.uncertainity_based_example_selector import UncertainityBasedExampleSelector

from activelearning.utils.validation import validate_input_table
from activelearning.utils.validation import validate_attr

class EntropyBasedExampleSelector(UncertainityBasedExampleSelector):
    
    def __init__(self):
        super(EntropyBasedExampleSelector, self).__init__()
    
    def _compute_entropy(self, probability):
        if 0 in probability:
            return 0
        else:
            return pd.np.sum(-probability * pd.np.log(probability))
    
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

        feature_values = unlabeled_dataset[feature_attrs]
        feature_values.fillna(value=0, inplace=True)
        # compute the prediction probabilities for the unlabeled dataset
        #start_time_predict_model = time.time()
        try:
            probabilities = model.predict_proba(feature_values) 
        except Exception as e:
            print e
            print feature_attrs
        #end_time_predict_model = time.time()
        #time_taken_predict_model = end_time_predict_model - start_time_predict_model
#             for v in unlabeled_dataset[feature_attrs].values: print v
#             print e
        # compute the entropy for the unlabeled pairs
        #entropies = {}
        
        #start_time_entropy_calc = time.time()
#         for i in xrange(len(probabilities)):
#             #print str(i) + " " + str(probabilities[i]) + "=>" + str(self._compute_entropy(probabilities[i]))
#             entropies[i] = self._compute_entropy(probabilities[i])
#         
        entropies = pd.np.sum(-probabilities * pd.np.log(probabilities), axis=1)
        
        #end_time_entropy_calc = time.time()
        #time_taken_entropy_calc = end_time_entropy_calc - start_time_entropy_calc
        
        entropies = dict(enumerate(entropies))
        
        
        #start_time_entropy_sort = time.time()
        candidate_examples = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(entropies))]
        #end_time_entropy_sort = time.time()
        #time_taken_entropy_sort = end_time_entropy_sort - start_time_entropy_sort
        
#         print "Time taken for calc prob"
#         print " time_taken_predict_model:" + str(time_taken_predict_model)
#         print " time_taken_entropy_calc:" + str(time_taken_entropy_calc)
#         print " time_taken_entropy_sort:" + str(time_taken_entropy_sort)
        next_batch_idxs = map(lambda val: val[0], candidate_examples)
        return unlabeled_dataset.iloc[next_batch_idxs]
