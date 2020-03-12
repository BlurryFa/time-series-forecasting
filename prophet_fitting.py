import numpy as np
from pandas import Series, DataFrame
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import *
import fbprophet

if __name__ == '__main__':
    df = data_reading()
    time_series = df['扬州市秦邮特种金属材料有限公司']
    index = time_series.index
    values = time_series.values
    df2 = DataFrame({'ds': index, 'y': values})
    df2_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05)


