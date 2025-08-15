from cbnn import *
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sksurv.preprocessing import OneHotEncoder
import unittest
import traceback


class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data=pd.read_csv('tests/time_varying_survival_data.csv').iloc[:,1:]
            train_data, val_data, test_data = split_data(data, test_size=0.05, val_size=0.15, random_state=None)
            # Get means, standard deviations (max_time is kept for plotting results later)
            sds = train_data.apply(np.std)
            means = train_data.apply(np.mean)
            max_time = train_data['time'].max().copy()
            time_var = 'time'
            status_var = 'status'
            
            train_data = normalizer(train_data, means, sds, max_time)
            val_data = normalizer(val_data, means, sds, max_time)
            test_data = normalizer(test_data, means, sds, max_time)
            
            # Remove status column, only include features
            test = test_data.drop(columns=[status_var])
            features = ['z1', 'z2', 'z3']
            time_var = 'time'
            event_var = 'status'
            
            cbnn_prep = prep_cbnn(features, train_data, time_var=time_var,
                                  event_var=event_var, ratio=100, comp_risk=False,
                                  layers=[5,5,5],device=device)
            cbnn_prep_val = sample_case_base(val_data, time=time_var,
                                             event=event_var, ratio=100,
                                             comprisk=False)
            cbnn_prep_val['offset']=pd.Series(cbnn_prep['offset'][0].repeat(cbnn_prep_val['offset'].shape[0]))
            fit=fit_hazard(cbnn_prep, epochs=20, batch_size=1000, val_data=cbnn_prep_val)
            cumulative_incidences,times=cu_inc_cbnn(fit, times=[time/100 for time in list(range(10,100,20))], x_test=test_data[features+[time_var]])
            shap_cbnn(fit, [time/100 for time in list(range(10,100,20))],val_data, x_test=test_data[features+[time_var]])
            survival_estimates_cbnn=1-cumulative_incidences
            #first column is array of time-estimates
        except Exception as e:
            # Capture the stack trace, including the line where the error occurred
            tb = traceback.format_exc()
            
            # Create a detailed error message
            error_message = f'The pipeline failed with exception: {str(e)}\nTraceback:\n{tb}'
            
            # Log or handle the failure (e.g., self.fail can be customized to log this appropriately)
            self.fail(error_message)
        

if __name__ == '__main__':
    unittest.main()