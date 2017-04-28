from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr

from activelearn.labeler.labeler import Labeler
from activelearn.exampleselector.example_selector import ExampleSelector

import time
class ActiveLearner(object):
    """
    A class which allows to learn a given model by actively querying the labels of unlabeled instances
    
    Args:
        model (Model): Model to learn
        example_selector (ExampleSelector): 
        labeler (Labeler): 
        batch_size (number): 
        num_iters: Number of iterations to run the active learner

    Attributes:
        model (Model): 
        example_selector (ExampleSelector): 
        labeler (string): 
        batch_size (boolean): Number of iterations to run the active learner
    """

    def __init__(self, model, example_selector, labeler, batch_size, num_iters):
        self.model = model
        self.example_selector = example_selector
        self.labeler = labeler
        self.batch_size = batch_size
        self.num_iters = num_iters
        self._labeled_dataset_ = None
        
    def learn(self, unlabeled_dataset, seed, exclude_attrs=None, context=None, 
              label_attr='label'):
        """
        Performs the Active Learning Loop to help learn the model by querying the labels of the instances
        
        Args:
            unlabeled_dataset (DataFrame): unlabeled_dataset
            seed (DataFrame): labeled examples
            
        Returns:
           A learned model  
        """
        
        #validate input tables
        validate_input_table(unlabeled_dataset, 'unlabeled dataset')
        validate_input_table(seed, 'seed')
        
        #validate labeler
        if not isinstance(self.labeler, Labeler):
            raise TypeError(self.labeler + ' is not an object of labeler class')
        #validate example selector
        if not isinstance(self.example_selector, ExampleSelector):
            raise TypeError(self.example_selector + ' is not an object of example selector ')
        
        feature_attrs = list(unlabeled_dataset.columns)  
        
        # remove exclude attrs from 
        feature_attrs = remove_exclude_attr(feature_attrs, exclude_attrs, unlabeled_dataset)
        labeled_dataset = seed
        i = 0
        #time_stats = []
        while i < self.num_iters:
            start_time_fitting_model = time.time()
            # train model using current set of labeled examples
            #print "#refitting the model"
            self.model = self.model.fit(labeled_dataset[feature_attrs].values,
                                        labeled_dataset[label_attr].values)

            #print "#selecting examples"
            # select current batch of examples to label
#             end_time_fitting_model = time.time()
#             time_fitting_model = end_time_fitting_model - start_time_fitting_model
#             
            #start_time_select_examples = time.time()
            selected_examples = self.example_selector.select_examples(unlabeled_dataset, 
                                                      self.model, exclude_attrs,
                                                      self.batch_size)
#             end_time_select_examples = time.time()
#             time_select_examples = end_time_select_examples - start_time_select_examples
            
            #time_stats.append((time_fitting_model,time_select_examples))
            #print "#label the examples"
            # label the selected examples
            labeled_examples = self.labeler.label(selected_examples, context, 
                                                  label_attr)

            unlabeled_dataset = unlabeled_dataset.drop(labeled_examples.index) 
            
            # update the labeled dataset
            labeled_dataset = labeled_dataset.append(labeled_examples) 
          
            i += 1
#         print "Time Fitting Model | Time Selecting Examples" 
#         for stat in time_stats:    
#             print str(stat[0]) +"    " + str(stat[1]) 
        self._labeled_dataset_ = labeled_dataset

        return self.model
