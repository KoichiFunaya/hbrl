# coding: utf-8
# pylint: disable = invalid-name, W0105, C0111, C0301
"""Scikit-learn wrapper interface for LightGBM."""
from __future__ import absolute_import

import os,sys
from sklearn import metrics
import _pickle as pickle

from lightgbm import LGBMClassifier

# The class below adds "score" function to the LGBMClassifer class
class LGBMClassifierWrapper(LGBMClassifier):
    """LightGBM Classifier Wrapper"""
    
    """
    Implementation of the scikit-learn API for LightGBMClassifier;
    The LightGBMClassifier class misses the score function.
    """

    def __init__(self,**kwargs):
        """
        initialize the wrapper class
        """
        LGBMClassifier.__init__(self,**kwargs)
        
        pass

    
    def score(self, X_test, y_test, **kwargs):
        """Return the score value for the test samples.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

            Test samples.
        
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
        
            True labels for X.

        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        score : float
        
            Mean accuracy of self.predict(X) wrt. y.

        """
        
        y_pred = self.predict(X_test)
        
        return metrics.accuracy_score(y_test,y_pred)

    
    def dump(self,file_name='object.pkl'):
        """Store the entire class object to a picle file.

        Parameters
        ----------
        file_name : file name
        
        Returns       
        -------

        """
        
        try:
            with open(file_name, "wb") as fp:   #Pickling
                pickle.dump(self, fp)
            return True
        
        except IOError as e:
            print('I/O error({0}): {1}'.format(e.errno, e.strerror))
            pass
        except ValueError:
            print('Could not convert data to an integer.')
            pass
        except:
            print('Unexpected error:{}'.format(sys.exc_info()[0]))
            raise
        
        return False
        
        
    def load(self,file_name='object.pkl'):
        """Store the entire class object to a picle file.

        Parameters
        ----------
        file_name : file name
        
        Returns       
        -------

        """
        
        try:
            with open(file_name, "rb") as fp:   # load the object from a pickle file
                self = pickle.load(fp)
            return True
        
        except IOError as e:
            print('I/O error({0}): {1}'.format(e.errno, e.strerror))
            pass
        except ValueError:
            print('Could not convert data to an integer.')
            pass
        except:
            print('Unexpected error:{}'.format(sys.exc_info()[0]))
            raise
        
        return False
    
    
