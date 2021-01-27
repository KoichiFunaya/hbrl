# coding: utf-8
# pylint: disable = invalid-name, W0105, C0111, C0301
"""Scikit-learn wrapper interface for LightGBM."""
from __future__ import absolute_import

import os,sys
import pandas as pd
import math
from parse import *
from sklearn import metrics
import _pickle as pickle
import getopt

from tqdm import tqdm
from lgbm_wrapper import LGBMClassifierWrapper,LGBMClassifierLoader
import lightgbm as lgbm

from ijcai import ExpBase


# The class below extracts element rules from an LGBM classifier
class LGBMRuleExtractor(ExpBase):
    """Rules extracted from Scikit-learn LightGBM Classifier"""
    
    """
    Extract rules from LGBMClassifierWrapper class objects,
    which is a collection of decision trees.
    The rules generated are a collection of single-leaf
    decision trees.
    """

    def __init__(self, booster=None, pickle_file=None, module_name='LGBMRuleExtractor', config_file=None, sub_module_name=None, args=('','')):
        """Creates a null list of trees.

        """
        
        if pickle_file:
            # Load self from a pickle file
            with open(pickle_file, 'rb') as pf:
                self = pickle.load(pf)
        
#        #LGBMClassifierWrapper.__init__(self, module_name, config_file, argv=('',''))
#        LGBMClassifierLoader(module_name="LGBMClassifierLoader", config_file=config_file, sub_module_name=sub_module_name)
        
        # initialize the ExpBase parent class
        ExpBase.__init__(self,module_name=module_name,config_file=config_file,sub_module_name=sub_module_name,args=args)

        # Read parameters from the config.toml
        self.model_file           = self.params['ember_datadir']['args']           + self.params['model_file']['args'] 
        self.rule_extractor_file  = self.params['rule_extractor_datadir']['args']  + self.params['rule_extractor_file']['args'] 
        self.antecedent_file      = self.params['antecedent_datadir']['args']      + self.params['antecedent_fname']['args'] 

        print("self.params = {}".format(self.params))
        print(self.model_file)
        print(self.rule_extractor_file)
        print(self.antecedent_file)

        # Initialize parser related memory
        #
        self.dicRoot   = {}
        self.listTrees = []
        self.dicImportantFeatures = {}
        self.dicParameters        = {}
        
        # Initialize the rules
        self.rules = []
        
        if booster:
            # load booster
            print('booster existed')
            self.booster = booster
        elif self.model_file:
            # Parse the model if provided
            if not os.path.exists(self.model_file):
                sys.exit("ember model {} does not exist".format(self.model_file))
            print('construct booster from model file')
            self.booster = lgbm.Booster(model_file=self.model_file)
        else:
            self.booster = None
            
        print("self.booster = {}".format(self.booster))
        
        if self.booster:
            self.parseLGBMModel(self.booster)
            self.add_pdNodes()
            self.add_pdLeaves()
            
        
        return

    
    def reset_parameters(self):
        """
        Reset parameters that can possibly be loaded.
        
        Parameters
        ----------

        Returns
        -------
                
        """
        
        self.params = dict()
        self.params['ember_datadir']          = {'opt':'--ember_datadir',          'args':''}
        self.params['model_file']             = {'opt':'--model_file',             'args':''}
        self.params['rule_extractor_datadir'] = {'opt':'--rule_extractor_datadir', 'args':''}
        self.params['rule_extractor_file']    = {'opt':'--rule_extractor_file',    'args':''}
        self.params['antecedent_datadir']     = {'opt':'--antecedent_datadir',     'args':''}
        self.params['antecedent_fname']       = {'opt':'--antecedent_fname',       'args':''}

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
        \t--ember_datadir <ember_datadir> \n\
        \t--model_file <model_file> \n\
        \t--rule_extractor_datadir <rule_extractor_datadir> \n\
        \t--rule_extractor_file <rule_extractor_file> \n\
        \t--antecedent_datadir <antecedent_datadir> \n\
        \t--antecedent_fname <antecedent_fname> \n\
         '.format(self.module_name))
        
        pass
    

    def load_arguments(self, args=("","")):
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
            elif opt in ("--ember_datadir"):          self.params['ember_datadir']['args']          = arg
            elif opt in ("--model_file"):             self.params['model_file']['args']             = arg
            elif opt in ("--rule_extractor_datadir"): self.params['rule_extractor_datadir']['args'] = arg
            elif opt in ("--rule_extractor_file"):    self.params['rule_extractor_file']['args']    = arg
            elif opt in ("--antecedent_datadir"):     self.params['antecedent_datadir']['args']     = arg
            elif opt in ("--antecedent_fname"):       self.params['antecedent_fname']['args']       = arg
            else                        :  pass

        # set the correct types for the parameters
        bool_types  = []
        int_types   = []
        float_types = []
        self.set_param_types(bool_types,int_types,float_types)
        
        if self.verbose > 20:
            for k in self.params:
                self.print_formated('{}{}'.format(k,type(self.params[k]['args'])),self.params[k]['args'])
        
        self.kwargs = {keys:self.params[keys]['args'] for keys in self.params.keys()}
        
        pass
    
    
    
    def parseLGBMModel(self,booster=None):
        """Parse LGBM model and store the trees and parameters in dictionaries
        
        Parameters
        ----------
        booster : LGBM model
        
        Return
        ------
        
        """
        # set the booster if provided
        if booster:
            self.booster = booster
        elif not self.booster:
            sys.exit()
        
        # Initialize the state
        state    = None

        # first extract lines from the LGBM model
        lines = self.booster.model_to_string().splitlines()
        
        # then parse the strings in the lines
        for s in lines:
            
            # First update the status
            if ('tree' in s) and ('=' not in s) and (':' not in s):
                state = 'root'
            elif 'Tree' in s:
                state = 'tree'
            elif 'feature importances:' in s:
                state = 'feature importances'
            elif 'parameters:' in s:
                state = 'parameters'
            
            # When state is not None, then parse the line
            if state:
                if 'root' in state:   # read parameters of the entire trees
                    t = parse("{}={}",s)
                    if t: self.dicRoot[t[0]] = t[1].split(' ')
                    elif 'tree' in s: pass
                    elif 'end of trees' in s: state = None
                    else:
                        state = None
                elif 'tree' in state: # parse parameters of the nodes and the leaves of trees
                    t = parse("{}={}",s)
                    if t:
                        if 'Tree' in t[0]:
                            dicTree            = {t[0]:t[1]}
                            dicNodes           = {}
                            dicLeaves          = {}
                        elif 'leaf' in t[0]:
                            dicLeaves[t[0]]    = t[1].split(' ')
                        elif 'num_leaves' in t[0]:
                            dicNodes['index']       = list(range(0,int(t[1])-1))
                            dicNodes['num_nodes']   = int(t[1])-1
                            dicLeaves['index']      = list(range(-1,-(int(t[1])+1),-1))
                            dicLeaves['num_leaves'] = int(t[1])
                        elif 'shrinkage' in t[0]:
                            dicNodes[t[0]]     = float(t[1])
                            dicLeaves[t[0]]    = float(t[1])
                        else:
                            dicNodes[t[0]]     = t[1].split(' ')
                    elif 'end of trees' in s: state = None
                    else:
                        dicTree['nodes']   = dicNodes
                        dicTree['leaves']  = dicLeaves
                        self.listTrees.append(dicTree)
                        state = None
                elif 'parameters' in state:
                    t = parse("[{}:{}]",s)
                    if t: self.dicParameters[t[0]]=t[1]
                    elif 'parameters' in s: pass
                    elif 'end of parameters' in s: state = None
                elif 'feature importances' in state:
                    t = parse("{}={}",s)
                    if t: self.dicImportantFeatures[t[0]]=t[1]
                    elif 'feature importances' in s: pass
                    
        return

    
    def add_pdLeaves(self):
        """convert leaf parameters into panda DataFrame and store them in the listTree.

        Input
        ----------
            
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        """
        
        # create an empty DataFrames
        LeafKeys = ['leaf_value','leaf_count']

        # fill the DataFrames
        for i in range(len(self.listTrees)):
            pdLeaves = pd.DataFrame(index=self.listTrees[i]['leaves']['index'])
            for k in LeafKeys:
                pdLeaves[k] = self.listTrees[i]['leaves'][k]
            self.listTrees[i]['leaves']['pdLeaves'] = pdLeaves
            
        return
    

    def add_pdNodes(self):
        """convert node parameters into panda DataFrame and store them in the listTree.

        Input
        ----------
            
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        listTrees  : list of trees with pdNodes DataFrames for each tree
        """
        
        # create an empty DataFrames
        NodeKeys = ['split_feature','threshold','split_gain',
                    'decision_type','left_child','right_child',
                    'internal_value','internal_count']

        # fill the DataFrames
        for i in range(len(self.listTrees)):
            pdNodes = pd.DataFrame(index=self.listTrees[i]['nodes']['index'])
            for k in NodeKeys:
                pdNodes[k] = self.listTrees[i]['nodes'][k]
            self.listTrees[i]['nodes']['pdNodes'] = pdNodes
            
        return
    

    def enum_children(self, pdNodes, pdLeaves, shrinkage, idx):
        """recursively walk the tree and extract rules
        
        Parameters
        ----------
        pdNodes : pandas DataFrame of the entire nodes in a tree
        pdLeaves: pandas DataFrame of the entire leaves in a tree
        idx     : current index in the DataFrame
        
        Return
        ------
        lRules  : list of all the child rules
        
        """
        
        if int(idx)>=0:
            # add a list on the left side of the node
            idx_left          = int(pdNodes.loc[idx,'left_child'])
            list_rules_left   = self.enum_children(pdNodes,pdLeaves,shrinkage,idx_left)
            
            # add a list on the right side of the node
            idx_right         = int(pdNodes.loc[idx,'right_child'])
            list_rules_right   = self.enum_children(pdNodes,pdLeaves,shrinkage,idx_right)

            # concatenate the two lists
            list_rules_concat = list_rules_left + list_rules_right
            
            # create an empty list
            list_rules        = []
            
            # add an node to all the lists
            while list_rules_concat:
                # first create a dictionary of the current node
                dic_new = {}
                dic_new['index']          = [idx]                             
                dic_new['split_feature']  = [pdNodes.loc[idx,'split_feature']]
                dic_new['threshold']      = [pdNodes.loc[idx,'threshold']]
                dic_new['split_gain']     = [pdNodes.loc[idx,'split_gain']] 
                dic_new['decision_type']  = [pdNodes.loc[idx,'decision_type']]
                dic_new['left_child']     = [idx_left]
                dic_new['right_child']    = [idx_right]
                dic_new['internal_value'] = [pdNodes.loc[idx,'internal_value']]
                dic_new['internal_count'] = [pdNodes.loc[idx,'internal_count']]
                dic_new['leaf_value']     = [None]
                dic_new['leaf_count']     = [None]
                dic_new['shrinkage']      = [shrinkage]
                
                # then pop a list, append the new dictionary to the list,
                # and then store in the resulting result
                dic = list_rules_concat.pop()
                for k in dic.keys():
                    dic[k] = dic_new[k] + dic[k]
                list_rules.append(dic)
        
        else:
            # if the node is already a leaf, then we assign a leaf in the list
            list_rules   = [{'index': [idx],
                             'leaf_value': [pdLeaves.loc[idx,'leaf_value']],
                             'leaf_count': [pdLeaves.loc[idx,'leaf_count']],
                             'split_feature': [None], 'threshold': [None], 'split_gain': [None],
                             'decision_type': [None], 'left_child': [None], 'right_child': [None],
                             'internal_value': [None], 'internal_count': [None],
                             'shrinkage':[shrinkage]}]

        return list_rules
    
    

    def induce_rules_from_a_tree(self, tree):
        """Induce rules from one decition tree.

        Parameters
        ----------
        tree: a tree containing nodes and leaves of the corresponding tree.
            
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        rules              : a list of rules contained in a one-leaf tree
        """

        # if listTrees is not in the argument, then use own one
        if not tree:
            sys.exit('Error: tree is not defined!')
        if 'pdNodes' not in tree['nodes'].keys():
            sys.exit('Error: pdNodes does not exist in the tree!')
        if 'pdLeaves' not in tree['leaves'].keys():
            sys.exit('Error: pdLeaves does not exist in the tree!')
        
        return self.enum_children(pdNodes  = tree['nodes']['pdNodes'],
                                  pdLeaves = tree['leaves']['pdLeaves'],
                                  shrinkage= tree['nodes']['shrinkage'],
                                  idx      = tree['nodes']['index'][0])
    
    
    def induce_rules(self):
        """Induce rules from multiple decition trees.

        Parameters
        ----------
            
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        rules              : a list of rules contained in all the trees
        """
        
        # induce the entire rules
        rules = None
        for tree in tqdm(self.listTrees):
            if rules:
                rules.extend(self.induce_rules_from_a_tree(tree))
            else:
                rules = self.induce_rules_from_a_tree(tree)
        
        # cast the induced rules into DataFrame
        for r in tqdm(rules):
            index  = r.pop('index')
            pdRule = pd.DataFrame(index=index)
            for k in r.keys():
                pdRule[k]=r[k]
                if 'child' in k:  # set a child to None if it is not visited next
                    for i in range(len(index)-1):
                        if index[i+1]!=pdRule.loc[index[i],k]:
                            pdRule.loc[index[i],k] = None
                    
            self.rules.append(pdRule)
        
        return self.rules
    

    def evaluateOneRule(self,ruleNr,X):
        """Add one line of rule.

        Parameters
        ----------
        ruleNr : the index in a panda DataFrame self.rules
        
        X      : a sample of feature vector
            
        **kwargs
            Other parameters for the prediction.

        Returns       
        -------
        match            : boolean value describing whether the sample matches the rule (TRUE) or not (False)
        shrinkage        : shrinkage
        leaf_value       : prediction value
        leaf_count       : number of samples observed in the training phase

        """
        
        pdRule   = self.rules[ruleNr]

        match = True
        
        for index in pdRule.index:
            feature = pdRule.loc[index,'split_feature']
            #print('left,right,leaf = {},{},{}'.format(pdRule.loc[index,'left_child'],
            #                                          pdRule.loc[index,'right_child'],
            #                                          pdRule.loc[index,'leaf_value']))
            
            if feature:
                column = 'Column_' + feature
                featureValue = float(X.loc[column])
                threshold    = float(pdRule.loc[index,'threshold'])
                if featureValue <= threshold:
                    if math.isnan(pdRule.loc[index,'left_child']): 
                        match = False
                    #printNodeEval(featureValue,threshold,'left_child',math.isnan(pdRule.loc[index,'left_child']),match)
                else:
                    if math.isnan(pdRule.loc[index,'right_child']): 
                        match = False
                    #printNodeEval(featureValue,threshold,'right_child',math.isnan(pdRule.loc[index,'right_child']),match)
            else:
                column = None
                leaf_value = float(pdRule.loc[index,'leaf_value'])
                if not pdRule.loc[index,'leaf_value']: 
                    match = False
                #printNodeEval(featureValue,threshold,'leaf',False,match)
                leaf_count = int(pdRule.loc[index,'leaf_count']) 
                shrinkage  = float(pdRule.loc[index,'shrinkage'])
                
        return match,shrinkage,leaf_value,leaf_count
    

    def drop_rules(self,support):
        """Drop rules according to the list 'support'.
        
        Parameters
        ----------
        support : a list consisting of boolean values, i.e. True or False
        
        Returns
        -------
        rules : list of rules that are left after removing those not supported.
        """
        
        try:
            assert(len(self.rules)==len(support))
        except:
            print('Length of the list \'support\' does not match that of rules')
            return self.rules

        # First convert two lists "self.fules" and "support" into one pandas dataframe.
        df = pd.DataFrame(self.rules,columns=['rule'])
        df['support'] = support
        
        # Then remove rows where the value of "support" column is "False".
        df = df[df.support==True]
        
        # Copy the "rule" column values back to the "self.rules"
        self.rules = df['rule'].tolist()
        
        return self.rules
    
    
    def save_rules(self,file_name='rule.pkl'):
        """Store rules to a picle file.

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
        
        
        
    def pickle(self,pickle_file=None):
        """Store rules to a picle file.

        Parameters
        ----------
        file_name : file name
        
        Returns       
        -------

        """
        # set the pickle file if None
        if not pickle_file:
            pickle_file = self.rule_extractor_file
            
        # dump this class object as a pickle file        
        try:
            with open(pickle_file, "wb") as fp:   #Pickling
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
        
        
    def extract_antecedent(self,cardinality=1,store=True,store_path=None):
        """
        Extract antecedent from the LGBM derived decision trees.
        Here, the antecedent is an assertion about the feature vector.
            E.g. (feature_649 > threshold_a) and (feature_34 < threshold_b)
        
        Parameters
        ----------
        cardinality: the number of assertions in the antecedent.
                     Currently we only support cardinality=1
        store      : if True, then store the extracted antecedents
                     in a pickle file.
        store_path : file path to store the pickle file.
                     if None, then read the file name from the config file.
        
        Returns       
        -------
        antecedent : list of pandas dataframes containing a list of antecedents.

        """
        
        # check that cardinality==1
        if cardinality != 1:
            print('cardinality > 1 is not supported.')
            return False
        
        # generate antecedents with cardinality==1
        antecedents = []
        for tree in self.listTrees:
            nodes = tree['nodes']['pdNodes']
            df    = pd.DataFrame(nodes[['split_feature','threshold']])
            antecedents.append(df)
            
        # store the rules in a pickle file
        if store:
            from utils import pickle_store
            if store_path:
                pickle_store(antecedents,store_path)
            else:
                pickle_store(antecedents,self.antecedent_file)
 
        return antecedents

        
def load_LGBMRuleExtractor(pickle_file='LGBMRuleExtractor'):

    # Load the class object from a pickle file
    with open(pickle_file, 'rb') as pf:
        rule_extractor = pickle.load(pf)
        
    return rule_extractor
        

        
def printNodeEval(featureValue,threshold,nodeType,isNan,match):
    
    print('fV,threshold,type,isnan,match= {:5.2f}, {:5.2f}, {}, {}, {}'.format(featureValue,
                                                                               threshold,
                                                                               nodeType,
                                                                               isNan,
                                                                               match))
    return

