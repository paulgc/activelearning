
class ActiveLearner(object):

    def __init__(self, model, example_selector, labeler, batch_size, num_iters):
        self.model = model
        self.example_selector = example_selector
        self.labeler = labeler
        self.batch_size = batch_size
        self.num_iters = num_iters
        self._labeled_dataset_ = None

    def learn(unlabeled_dataset, seed, exclude_attrs=None, context=None, 
              label_attr='label'):
       # remove exclude attrs                                                  
        feature_attrs = list(unlabeled_dataset.columns)                         
        if exclude_attrs:                                                       
            for attr in exclude_attrs:                                          
                feature_attrs.remove(attr)          

        labeled_dataset = seed
        i = 0

        while i < self.num_iters:
            # train model using current set of labeled examples
            self.model = self.model.fit(labeled_dataset[feature_attrs].values,
                                        labeled_dataset[label_attr].values)

            # select current batch of examples to label
            selected_examples = self.example_selector(unlabeled_dataset, 
                                                      self.model, exclude_attrs,
                                                      self.batch_size)

            # label the selected examples
            labeled_examples = self.labeler.label(selected_examples, context, 
                                                  label_attr)

            # remove labeled examples from the unlabeled dataset 
            # TODO: NOT SURE IF THIS WILL WORK CORRECTLY. TEST THIS
            # TODO: NOT SURE IF THIS WILL WORK CORRECTLY. TEST THIS
            # TODO: NOT SURE IF THIS WILL WORK CORRECTLY. TEST THIS
            # TODO: NOT SURE IF THIS WILL WORK CORRECTLY. TEST THIS                    
            unlabeled_dataset = unlabeled_dataset.drop(labeled_examples.index) 

            # update the labeled dataset
            labeled_dataset = labeled_dataset.append(labeled_examples) 
          
            i += 1

        self._labeled_dataset_ = labeled_dataset

        return self.model
