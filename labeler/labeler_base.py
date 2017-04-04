'''
Created on Mar 7, 2017

@author: lokananda
'''


class LabelerBase(object):

    
    def __init__(self, label_names):
        self.label_names = label_names
        
    def get_label(self, feature):
        """Queries and returns the class labels for the input feature array.

        Parameters
        ----------
        feature : array-like, shape (n_features,)
            The feature vector whose label is to queried.

        Returns
        -------
        label : int
            The class label of the queried feature.
        """
        pass
    
    
    