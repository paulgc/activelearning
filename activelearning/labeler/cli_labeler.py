
from labeler import Labeler

class CliLabeler(Labeler):
    
    def __init__(self, get_instruction_fn, get_example_display_fn, labels):
        self.get_instruction_fn = get_instruction_fn
        self.get_example_display_fn = get_example_display_fn
        self.labels = labels
 
    def _input_from_stdin(self, banner_str):
        return raw_input(banner_str)
    
    def label(self, examples_to_label, context, label_attr='label'):
        print(self.get_instruction_fn(context))

        user_labels = []
        for idx, example in examples_to_label.iterrows():
            label_str = self._input_from_stdin(
                            self.get_example_display_fn(example, context))
            user_labels.append(self.labels[label_str]) 

        examples_to_label[label_attr] = user_labels
        return examples_to_label