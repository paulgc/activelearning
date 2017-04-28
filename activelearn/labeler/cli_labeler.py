from activelearn.labeler.labeler import Labeler
from activelearn.utils.validation import validate_input_table

class CliLabeler(Labeler):
    """
    A command line labeler for labeling raw instances
    
    Args:
        get_instruction_fn (Function): User provided function which specifies the instruction is to be displayed to the user
        get_example_display_fn (Function): User provided function which specifies how to fetch the raw examples to be labeled 
    Attributes:
        get_instruction_fn (Function)
        get_example_display_fn (Function)
    """
    def __init__(self, get_instruction_fn, get_example_display_fn, labels):
        self.get_instruction_fn = get_instruction_fn
        self.get_example_display_fn = get_example_display_fn
        self.labels = labels
 
    def _input_from_stdin(self, banner_str):
        return raw_input(banner_str)
    
    def validate_label_input(self, raw_label_str):
        if self.labels.has_key(raw_label_str):
            return True
        else:
            return False
        
    def label(self, examples_to_label, context, label_attr='label'):
        
        # check if the input examples_to_label is a dataframe
        validate_input_table(examples_to_label, 'unlabeled dataset')
        
        #Show the instruction to the user 
        print(self.get_instruction_fn(context))
        
        user_labels = []
        
        for example in examples_to_label.iterrows():
            #Fetch and display the raw example to be labeled
            label_str = self._input_from_stdin(
                            self.get_example_display_fn(example, context))
            if self.validate_label_input(label_str):
                user_labels.append(self.labels[label_str])
            else:
                #Display error message if user enters a wrong label
                print("Incorrect Label. Pls try again")
                self.label(examples_to_label, context, label_attr='label')
            

        examples_to_label[label_attr] = user_labels
        return examples_to_label