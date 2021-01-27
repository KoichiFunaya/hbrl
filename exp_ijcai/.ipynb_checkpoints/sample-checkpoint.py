# coding: utf-8

import os,sys,pathlib
ROOT_PATH = pathlib.Path.cwd().parent.resolve().as_posix()
sys.path.insert(0,ROOT_PATH)

import pymc3 as pm
from pymc3 import Continuous, Discrete, Multinomial, find_MAP
from pymc3.model import Model
import theano.tensor as tt

from hbrl_hmc import HBRL_L1Prior,HBRL_L1Likelihood



def sample_HBRL_L1(X,y,dic,s_prev=None):
    
    # set the parameters
    K           = dic['K']
    N           = dic['N']
    L           = dic['L']
    _lambda_    = dic['_lambda_']
    L0_pos      = dic['L0_pos']
    s_prev      = dic['s_prev']
    _zeta_      = dic['_zeta_']
    _mu_        = dic['_mu_']
    _gamma_     = dic['_gamma_'] 
    _tau_       = dic['_tau_']
    _xi_        = dic['_xi_']
    _kappa_     = dic['_kappa_']
    _nu_        = dic['_nu_']
    _eta_       = dic['_eta_']
    _rho_       = dic['_rho_']
    min_capture = dic['min_capture']
    K_target    = dic['K_target']
    nSample     = dic['nSample']
    nTune       = dic['nTune']
    nCores      = dic['nCores']
    

    # filter samples down to target size
    Xn,yn,idx_conditions,idx_samples,Kn,Nn,Ln,n_prev = filter_samples(X=X,y=y,s_prev=s_prev,min_capture=dic['min_capture'],K_target=dic['K_target'])
    
    # Create the HBRL model
    with pm.Model(name="HBRL_L1") as model:
        s = HBRL_L1Prior("s",X=Xn,l0_pos=n_prev,_lambda_=_lambda_,_zeta_=_zeta_,_gamma_=_gamma_,_tau_=_tau_,_xi_=_xi_,_kappa_=_kappa_,_nu_=_nu_)
        y = HBRL_L1Likelihood("y",s=s,X=Xn,_eta_=_eta_,_xi_=_xi_,_kappa_=_kappa_,_nu_=_nu_,_rho_=_rho_,observed=yn,shape=(L,N))
        
    # initialize NUTS sampler
    with model:
        start,nuts_sampler = pm.sampling.init_nuts(init='auto', chains=1, progressbar=True)
        nuts_sampler.max_treedepth = 15
        nuts_sampler.target_accept = 0.999
        
    # Create a place holder dictionary
    result = {}
    
    # MCMC Sampling using Metropolis Hasting
    with model:
        step1 = pm.Metropolis()
        #step2 = pm.NUTS(target_accept=0.97)
        step2 = nuts_sampler
        result["trace"] = pm.sample(sample_count, init='auto', tune=nTune, cores=nCores, step=[step1,step2], start=start[-1])

    # store the models
    result["model"]             = model
    result["HBRL_L1Prior"]      = HBRL_L1Prior
    result["HBRL_L1Likelihood"] = HBRL_L1Likelihood
    
    
    return result