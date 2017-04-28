from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr

from activelearn.labeler.labeler import Labeler
from activelearn.exampleselector.example_selector import ExampleSelector

import time
class ActiveLearner(object):
    """
    A class which allows to learn a given model by actively querying the labels
    of unlabeled instances using Pool-based active learning.
    
    Args:
        model (Model): Model to learn
        example_selector (ExampleSelector): example selector to query 
					    informative examples
        labeler (Labeler): A labeler which fetches labels from the oracle
        batch_size (number): The number of examples to be labeled per iteration
        num_iters (number): Number of iterations to run the active learner

    Attributes:
        model (Model): Model to learn
        example_selector (ExampleSelector): example selector to query 
					    informative examples
        labeler (Labeler): A labeler which fetches labels from the oracle
        batch_size (number): The number of examples to be labeled per iteration
        num_iters (number): Number of iterations to run the active learner
    """

    def __init__(self, model, example_selector, labeler, batch_size, num_iters):
        self.model = model
        self.example_selector = example_selector
        self.labeler = labeler
        self.batch_size = batch_size
        self.num_iters = num_iters
        self._labeled_dataset_ = None
        
    def learn(self, unlabeled_dataset, seed, exclude_attrs=None, context=None, 
              label_attr='label', not_sure_label_allowed=False):
        """
        Performs the Active Learning Loop to help learn the model by querying 
	the labels of the instances
        
        Args:
            unlabeled_dataset (DataFrame): A Datframe containing unlabeled
					   examples
            
	    seed (DataFrame): A Dataframe containing initial labeled examples
			      which is used to learn the initial model
            
	    exclude_attrs (list): A list of attributes to be excluded while
				  fitting the model (Defaults to None)
	    
	    context (map): An user defined map containing all the neccessary 
			object that is used by the user to query the raw 
			representation of examples.
	  
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
        
        # remove exclude attrs from feature attrs
        feature_attrs = remove_exclude_attr(feature_attrs, exclude_attrs, unlabeled_dataset)
        labeled_dataset = seed
        i = 0

        while i < self.num_iters:

            # train model using current set of labeled examples
            self.model = self.model.fit(labeled_dataset[feature_attrs].values,
                                        labeled_dataset[label_attr].values)

            selected_examples = self.example_selector.select_examples(unlabeled_dataset, 
                                                      self.model, exclude_attrs,
                                                      self.batch_size)
            # label the selected examples
            labeled_examples = self.labeler.label(selected_examples, context, 
                                                  label_attr)
            
            # drop the labeled examples from the pool of unlabled examples
            unlabeled_dataset = unlabeled_dataset.drop(labeled_examples.index) 
            
            # update the labeled dataset
            labeled_dataset = labeled_dataset.append(labeled_examples) 
          
            i += 1

        self._labeled_dataset_ = labeled_dataset

        return self.model
