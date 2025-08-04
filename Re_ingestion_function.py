# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 22:45:08 2025

@author: jec822
"""

from utils import connect_to_dj
from schemas import twop_opto_analysis
from schemas import spont_timescales
from ingest_functions import ingest2PoptoAnalysis
from ingest_functions import ingestTau

import warnings
warnings.filterwarnings("ignore")


# %%

rec_list = ingest2PoptoAnalysis.getDefaultTwopOptoList()

ingestTau.ingestTwopTauSession_batch(rec_list,tau_param_set_id=2)
# %%

rec_list = ingest2PoptoAnalysis.getDefaultTwopOptoList()
for trig_id in list([2,4]):
    for incl_id in list([9]):
        print('')
        print('CORR {}, INCL {}'.format(trig_id,incl_id))
        ingest2PoptoAnalysis.ingestTwopOptoSession_batch(rec_list,trigdff_param_set_id=trig_id, trigdff_inclusion_param_set_id=incl_id)
        
        
        
