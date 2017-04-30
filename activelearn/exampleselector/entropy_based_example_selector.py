
import operator
import pandas as pd

from activelearn.exampleselector.uncertainty_based_example_selector import UncertaintyBasedExampleSelector

from activelearn.utils.validation import validate_input_table
from activelearn.utils.validation import validate_attr
from activelearn.utils.helper_functions import remove_exclude_attr
from six.moves import xrange

class EntropyBasedExampleSelector(UncertaintyBasedExampleSelector):
    """
    Entropy based Uncertainty example selection
    """
    def __init__(self):
        super(EntropyBasedExampleSelector, self).__init__()
    
    def _utility_example(self, probability):
        self._compute_entropy(probability)
    
    def _compute_entropy(self, probability):
        if 0 in probability:
            return 0
        else:
            return pd.np.sum(-probability * pd.np.log(probability))
    
    def select_uncertain_examples(self, probabilities, unlabeled_dataset, batch_size):
        """
        Used to select informative examples based on the entropy of the examples.

        Args:
            model (Model): Model that is used to compute the uncertainty measure of the example
            unlabeled_dataset (Pandas DataFrame): A Dataframe containing unlabeled examples
            exclude_attrs (list): Attributes which are not feature attributes.
            batch_size (number): The number of examples to select
        
        Returns:
            The table of most informative examples to be labeled (DataFrame)
        """


        entropies = {} 
        # compute the entropy for the unlabeled pairs
        for i in xrange(len(probabilities)):
            entropies[i] = self._compute_entropy(probabilities[i])
        
        candidate_examples = sorted(entropies.items(), key=operator.itemgetter(1), reverse=True)[:min(batch_size, len(entropies))]

        next_batch_idxs = [val[0] for val in candidate_examples]
        return unlabeled_dataset.iloc[next_batch_idxs]
