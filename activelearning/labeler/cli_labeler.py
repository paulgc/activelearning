
from labeler import Labeler

class CliLabeler(Labeler):
    
    def __init__(self, get_instruction_fn, get_example_display_fn, labels):
        self.get_instruction_fn = get_instruction_fn
        self.get_example_display_fn = get_example_display_fn
        self.labels = labels
 
    def _input_from_stdin(self, banner_str):
        return raw_input(banner_str)
    
    def validate_input(self, raw_label_str):
        if self.labels.has_key(raw_label_str):
            return True
        else:
            return False
        
    def label(self, examples_to_label, context, label_attr='label'):
        #Show the instruction to the user 
        print(self.get_instruction_fn(context))
        user_labels = []
        for idx, example in examples_to_label.iterrows():
            label_str = self._input_from_stdin(
                            self.get_example_display_fn(example, context))
            if self.validate_input(label_str):
                user_labels.append(self.labels[label_str])
            else:
                #Display error message if user enters a wrong label
                print "Incorrect Label. Pls try again"
                self.label(examples_to_label, context, label_attr='label')
            

        examples_to_label[label_attr] = user_labels
        return examples_to_label