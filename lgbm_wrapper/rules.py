# coding: utf-8
# pylint: disable = invalid-name, W0105, C0111, C0301
"""Rules extracted from Scikit-learn LightGBM Classifier"""
from __future__ import absolute_import

class LGBMRules(object):
    """Rules extracted from Scikit-learn LightGBM Classifier"""
    
    """
    load rules generated with LGBMRuleExtractor class objects,
    which extracts rules from LGBM Classifer.
    The rules loaded are a collection of single-leaf
    decision trees, derived from the original tress in LGBM Classifier.
    """

    def __init__(self, rule_file=None):
        """
        Initialize the parameters.
        Load rules from a file
        """
        
        # Initialize the rules
        self.rules = []
        
        # load rules if rule_file exhists
        if rule_file:
            return self.load(file_name=rule_file)
        
        # Otherwise return null list
        else:
            return []
    

    def load_rules(self,file_name='rule.pkl'):
        """Load rules from a pickle file.

        Parameters
        ----------
        file_name : file name
        
        Returns       
        -------

        """
        
        try:
            with open(file_name, "rb") as fp:   #Pickling
                self.rules = pickle.load(self.rules, fp)
            return self.rules
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
        
        
        
    def save_rules(self,file_name='rule.pkl'):
        """save rules in a picle file.

        Parameters
        ----------
        file_name : file name
        
        Returns       
        -------

        """
        
        try:
            with open(file_name, "wb") as fp:   #Pickling
                pickle.dump(self.rules, fp)
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
