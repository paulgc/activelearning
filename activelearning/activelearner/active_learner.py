from activelearning.utils.validation import validate_input_table
from activelearning.utils.validation import validate_attr
import time
class ActiveLearner(object):

    def __init__(self, model, example_selector, labeler, batch_size, num_iters):
        self.model = model
        self.example_selector = example_selector
        self.labeler = labeler
        self.batch_size = batch_size
        self.num_iters = num_iters
        self._labeled_dataset_ = None

    def remove_exclude_attr(self, feature_attrs, exclude_attrs, dataset):
        for attr in exclude_attrs:
            validate_attr(attr, dataset.columns, "attr", 'unlabeled_dataset')
                                    
        if exclude_attrs:                                                       
            for attr in exclude_attrs:                                          
                feature_attrs.remove(attr) 
        
        return feature_attrs
        
    def learn(self, unlabeled_dataset, seed, exclude_attrs=None, context=None, 
              label_attr='label'):
        
        validate_input_table(unlabeled_dataset, 'unlabeled dataset')
        validate_input_table(seed, 'seed')
        feature_attrs = list(unlabeled_dataset.columns)  
        
        # remove exclude attrs from 
        self.remove_exclude_attr(feature_attrs, exclude_attrs, unlabeled_dataset)
        labeled_dataset = seed
        i = 0
        time_stats = []
        while i < self.num_iters:
            start_time_fitting_model = time.time()
            # train model using current set of labeled examples
            #print "#refitting the model"
            self.model = self.model.fit(labeled_dataset[feature_attrs].values,
                                        labeled_dataset[label_attr].values)

            #print "#selecting examples"
            # select current batch of examples to label
            end_time_fitting_model = time.time()
            time_fitting_model = end_time_fitting_model - start_time_fitting_model
            
            start_time_select_examples = time.time()
            selected_examples = self.example_selector.select_examples(unlabeled_dataset, 
                                                      self.model, exclude_attrs,
                                                      self.batch_size)
            end_time_select_examples = time.time()
            time_select_examples = end_time_select_examples - start_time_select_examples
            
            time_stats.append((time_fitting_model,time_select_examples))
            #print "#label the examples"
            # label the selected examples
            labeled_examples = self.labeler.label(selected_examples, context, 
                                                  label_attr)

            # remove labeled examples from the unlabeled dataset 
            # TODO: NOT SURE IF THIS WILL WORK CORRECTLY. TEST THIS
#             print labeled_examples
#             print labeled_examples.index
            
            unlabeled_dataset = unlabeled_dataset.drop(labeled_examples.index) 
            
            labeled_examples.fillna(value=0, inplace=True)
            # update the labeled dataset
            labeled_dataset = labeled_dataset.append(labeled_examples) 
          
            i += 1
#         print "Time Fitting Model | Time Selecting Examples" 
#         for stat in time_stats:    
#             print str(stat[0]) +"    " + str(stat[1]) 
        self._labeled_dataset_ = labeled_dataset

        return self.model
