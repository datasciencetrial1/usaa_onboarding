from sklearn.externals import joblib
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
 
clf = joblib.load('./model/XGBoost_CreditCard_GridCV_45_Iter_v1.pkl')
DUMMY_VALUES = {u'EDUCATION': [2, 1, 3, 5, 4, 6, 0], u'MARRIAGE': [2, 1, 3, 0], u'SEX': [2, 1]}

def dummy_encode_dataframe(df):
    for (feature, dummy_values) in DUMMY_VALUES.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, str(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]

def predict_credit_card_default(LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6):
    new_list = [[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]]
    new_df = pd.DataFrame(new_list, columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
    
    # Dummy encode
    dummy_encode_dataframe(new_df)
    
    # Recale numerical features
    new_df['LIMIT_BAL'] = (new_df['LIMIT_BAL'] - 165498.715780).astype(np.float64) / 129130.743065
    new_df['AGE'] = (new_df['AGE'] - 35.380849).astype(np.float64) / 9.271046
    new_df['PAY_0'] = (new_df['PAY_0'] + 0.001667).astype(np.float64) / 1.127136
    new_df['PAY_2'] = (new_df['PAY_2'] + 0.123463).astype(np.float64) / 1.200591
    new_df['PAY_3'] = (new_df['PAY_3'] + 0.154756).astype(np.float64) / 1.204058
    new_df['PAY_4'] = (new_df['PAY_4'] + 0.211675).astype(np.float64) / 1.166573
    new_df['PAY_5'] = (new_df['PAY_5'] + 0.252885).astype(np.float64) / 1.137006
    new_df['PAY_6'] = (new_df['PAY_6'] + 0.278011).astype(np.float64) / 1.158191
    new_df['BILL_AMT1'] = (new_df['BILL_AMT1'] - 50598.928663).astype(np.float64) / 72650.197809
    new_df['BILL_AMT2'] = (new_df['BILL_AMT2'] - 48648.047418).astype(np.float64) / 70365.395642
    new_df['BILL_AMT3'] = (new_df['BILL_AMT3'] - 46368.903537).astype(np.float64) / 68194.719520
    new_df['BILL_AMT4'] = (new_df['BILL_AMT4'] - 42369.872828).astype(np.float64) / 63071.455167
    new_df['BILL_AMT5'] = (new_df['BILL_AMT5'] - 40002.333097).astype(np.float64) / 60345.728280
    new_df['BILL_AMT6'] = (new_df['BILL_AMT6'] - 38565.266636).astype(np.float64) / 59156.501143
    new_df['PAY_AMT1'] = (new_df['PAY_AMT1'] - 5543.098046).astype(np.float64) / 15068.862730
    new_df['PAY_AMT2'] = (new_df['PAY_AMT2'] - 5815.529).astype(np.float64) / 20797.44
    new_df['PAY_AMT3'] = (new_df['PAY_AMT3'] - 4969.431393).astype(np.float64) / 16095.929295
    new_df['PAY_AMT4'] = (new_df['PAY_AMT4'] - 4743.656861).astype(np.float64) / 14883.554872
    new_df['PAY_AMT5'] = (new_df['PAY_AMT5'] - 4783.643693).astype(np.float64) / 15270.703904
    new_df['PAY_AMT6'] = (new_df['PAY_AMT6'] - 5189.573607).astype(np.float64) / 17630.718575
    
    result = clf.predict_proba(new_df)
    return result.tolist()