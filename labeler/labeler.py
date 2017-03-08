'''
Created on Mar 7, 2017

@author: lokananda
'''


class Labeler(object):

    """Label the queries made by ExampleSelector

    Assign labels to the samples queried by ExampleSelector.
    """
    def label(self, feature):
        """Return the class labels for the input feature array.

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
    
    
    