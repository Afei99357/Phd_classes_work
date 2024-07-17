#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re


start = pd.read_csv("onlinefraud.csv")
# start.columns == [
#    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
#    'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
#    'isFlaggedFraud']


type_code = np.unique(start['type'])
start["type"] = np.searchsorted(type_code, start["type"])
start["nameOrig"] = [int(x[1:]) for x in start["nameOrig"]]

dest_code = np.unique([t[0] for t in start['nameDest']])
start["dest"] = np.searchsorted(dest_code, [t[0] for t in start['nameDest']])
start["nameDest"] = [int(x[1:]) for x in start["nameDest"]]
column_names = start.columns
start = start.to_numpy()


np.savez_compressed(
    "onlinefraud.npz",
    fraud=start,
    column_names=column_names
)
