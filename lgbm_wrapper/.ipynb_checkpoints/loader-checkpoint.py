# coding: utf-8
# pylint: disable = invalid-name, W0105, C0111, C0301
from __future__ import absolute_import

import os,sys
import getopt
from ijcai import ExpBase

from .wrapper import LGBMClassifierWrapper

# The class below loads the LGBMClassiferLoader class object
class LGBMClassifierLoader(ExpBase):
    """LightGBM Classifier Loader"""
    
    """
    Loads LightGBMClassifier class objects.
    """

    def __init__(self, module_name, config_file, sub_module_name="LGBMClassifierLoader", create=True, args=("","")):
        """
        initialize the wrapper class
        """
        # initialize the ExpBase parent class
        ExpBase.__init__(self,module_name=module_name,config_file=config_file,sub_module_name=sub_module_name,args=args)

        # create the class object
        if create: self.LGBMClassifierWrapper = LGBMClassifierWrapper(**self.kwargs)
        else:      self.LGBMClassifierWrapper = None
        
        pass

    
    def reset_parameters(self):
        """
        Reset parameters that can possibly be loaded.
        
        Parameters
        ----------

        Returns
        -------
                
        """
        
        self.params = dict()
        self.params['boosting']         = {'opt':'-b', 'args':''}
        self.params['objective']        = {'opt':'-j', 'args':''}
        self.params['num_iterations']   = {'opt':'-y', 'args':''}
        self.params['learning_rate']    = {'opt':'-l', 'args':''}
        self.params['num_leaves']       = {'opt':'-q', 'args':''}
        self.params['seed']             = {'opt':'-s', 'args':''}
        self.params['max_depth']        = {'opt':'-x', 'args':''}
        self.params['min_data_in_leaf'] = {'opt':'-y', 'args':''}
        self.params['feature_fraction'] = {'opt':'-y', 'args':''}

        pass
    
    
    def print_usage(self):
        """ 
        Print the usage of this module
    
        Parameters
        ----------

        Returns
        -------
                
        """
        
        print('usage: {}.py \n\
        \t--boosting <boosting method> \n\
        \t--objective <objective> \n\
        \t--num_iterations <# of iterations> \n\
        \t--learning_rate <learning rate> \n\
        \t--num_leaves <number of leaves> \n\
        \t--seed <seed to generate other random number seeds> \n\
        \t--max_depth <maximum tree depth> \n\
        \t--min_data_in_leaf <minimu data in leaf nodes> \n\
        \t--feature_fraction <feature fraction> \n\
        '.format(self.module_name))
        
        pass
    

    def load_parameters(self, args=("","")):
        """
        Load parameters from the function arguments.
        
        Parameters
        ----------

        Returns
        -------
                
        """
        
        # if file paths and parameters are specified in the argument,
        # then replace the default values with them.
        # You cannot use the option flags that matches the first letter of 2+ parameter names.
        # This means these option flags are NG: e,f,o,m,n
        try:
            opts, _args = getopt.getopt(args,"hb:j:l:q:s:x:y:",
                                       ["boosting=","objective=","learning_rate=","num_leaves=","seed=","max_depth=","min_data_in_leaf="])
        except getopt.GetoptError:
            print('args={}'.format(args))
            self.print_usage()
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print_usage()
                sys.exit()
            elif opt in ("--boosting"):         self.params['boosting']['args']         = arg
            elif opt in ("--objective"):        self.params['objective']['args']        = arg
            elif opt in ("--num_iterations"):   self.params['num_iterations']['args']   = arg
            elif opt in ("--learning_rate"):    self.params['learning_rate']['args']    = arg
            elif opt in ("--num_leaves"):       self.params['num_leaves']['args']       = arg
            elif opt in ("--seed"):             self.params['seed']['args']             = arg
            elif opt in ("--max_depth"):        self.params['max_depth']['args']        = arg
            elif opt in ("--min_data_in_leaf"): self.params['min_data_in_leaf']['args'] = arg
            elif opt in ("--feature_fraction"): self.params['feature_fraction']['args'] = arg
            else                        :  pass
            # there could be other options, but simply pass on to the 

        # set the correct types for the parameters
        bool_types  = []
        int_types   = ['num_iterations','num_leaves','max_depth','min_data_in_leaf','seed']
        float_types = ['learning_rate','feature_fraction']
        self.set_param_types(bool_types,int_types,float_types)
        
        if self.verbose > 20:
            for k in self.params:
                self.print_formated('{}{}'.format(k,type(self.params[k]['args'])),self.params[k]['args'])
        
        self.kwargs = {keys:self.params[keys]['args'] for keys in self.params.keys()}
        
        pass
    
