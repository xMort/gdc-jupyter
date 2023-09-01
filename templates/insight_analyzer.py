import os
from dotenv import load_dotenv
from gooddata_pandas import GoodPandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import Birch
from adtk.data import validate_series
from adtk.detector import PersistAD
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import simplejson as json


class InsightAnalyzer:

    def __init__(self, insight_id: str, workspace: str | None = None):
        load_dotenv()
        self.host = os.getenv("HOST")
        self.token = os.getenv("TOKEN")
        self.workspace_id = os.getenv("WORKSPACE") if workspace is None else workspace
        self.insight_id = insight_id

    def _fetch_data(self):
        gp = GoodPandas(host=self.host, token=self.token)
        return gp.data_frames(self.workspace_id).for_insight(self.insight_id)

    def describe_data(self):
        df = self._fetch_data()
        return df.describe()

    def show_points(self):
        df = self._fetch_data()
        plt.scatter(df[df.columns[1]], df[df.columns[0]])
        plt.show()

    def cluster(self, cluster_count: int = 3):
        df = self._fetch_data()
        X = np.column_stack((df[df.columns[1]], df[df.columns[0]]))
        model = Birch(threshold=0.01, n_clusters=cluster_count)
        yhat = model.fit_predict(X)

        for cluster in np.unique(yhat):
            plt.scatter(X[yhat == cluster, 0], X[yhat == cluster, 1])
        plt.show()

    @staticmethod
    def _prepare_train_test_data(df):
        steps = round(len(df) * 0.1)
        data_train = df[:-steps]
        data_test = df[-steps:]
        return data_train, data_test

    def show_train_data(self):
        df = self._fetch_data().asfreq('H')
        data_train, data_test = self._prepare_train_test_data(df)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        data_train[df.columns[0]].plot(ax=ax, label='train')
        data_test[df.columns[0]].plot(ax=ax, label='test')
        ax.legend()

    def anomalies(self):
        df = self._fetch_data().asfreq('H')
        data = df[df.columns[0]].interpolate(method='linear')
        validate_series(data)

        seasonal_ad = PersistAD(c=3, side="both")
        anomalies = seasonal_ad.fit_detect(data)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        data.plot(ax=ax, label='train')
        anomalies = anomalies.fillna(False)

        # Filter the data using the anomalies binary mask to get the anomaly values.
        anomaly_values = data[anomalies]

        # Use scatter to plot the anomalies as points.
        ax.scatter(anomaly_values.index, anomaly_values, color='red', label='anomalies')

        ax.legend()


    def predict(self):
        
        df = self._fetch_data()
        df = df.asfreq('H')

        # number of steps we will try to predict
        steps = round(len(df) * 0.1)
        data_train = df[:-steps]
        data_test = df[-steps:]
        dim_y = df.columns.values[0]
        # Seed for reproductibility
        seed = 123
        # number of periods to use as a predictor for the next one
        lags = 168
        forecaster = ForecasterAutoreg(
            regressor=RandomForestRegressor(random_state=seed),
            lags=lags,
        )


        dim_y = df.columns.values[0]
        data_interpolate = data_train.copy()
        data_interpolate.loc[:, dim_y] = data_interpolate.loc[:, dim_y].interpolate(method='linear')


        forecaster.fit(y=data_interpolate[dim_y])

        
        predictions = forecaster.predict(steps=steps)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        data_interpolate[dim_y].plot(ax=ax, label='train')
        data_test[dim_y].plot(ax=ax, label='test')
        predictions.plot(ax=ax, label='predictions')
        ax.legend()

        interpolated_data = df[dim_y].interpolate(metho='linear')
        result = pd.merge(
            predictions.to_frame(),
            interpolated_data.rename('real'),
            left_index=True,
            right_index=True,
            how='outer'
            ).reset_index().to_dict()
        
        for key, value in result['index'].items():
            result['index'][key] = str(value)
        def handle_nan(o):
            if np.isnan(o): 
                return None 
            return o
        return json.dumps(result, ignore_nan=True)

    def predict_bounds(self, confidence_interval=0.95, steps=50):
        # Step 1: Interpolate missing values
        df = self._fetch_data().asfreq('H')
        column_name = df.columns[0]
        steps = round(len(df) * 0.1)
        df_train = df[:-steps]
        interpolated_series = df_train[column_name].interpolate()

        # Step 2: Fit SARIMAX model 
        model = SARIMAX(interpolated_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)

        # Step 3: Forecast next steps
        forecast = fit.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        
        # Calculate the confidence intervals
        conf_int = forecast.conf_int(alpha=(1-confidence_interval))
        conf_int_lower = conf_int.iloc[:, 0]
        conf_int_upper = conf_int.iloc[:, 1]
        
        # Create results DataFrame
        result_df = pd.DataFrame({
            'prediction': mean_forecast,
            'lower_bound': conf_int_lower,
            'upper_bound': conf_int_upper
        })
        
        # Extract only interpolated values for visualization
        interpolated_only = interpolated_series.copy()
        null_intervals = df[column_name].isnull()
        # Mask the interpolated series to show only where values were interpolated
        interpolated_only[~null_intervals] = np.nan


        df_real = df[-steps:]
        # Plotting
        plt.figure(figsize=(10, 4))
        df_train[column_name].plot(label='Train Data', color='blue')
        df_real[column_name].plot(label='Real Data', color='orange')
        interpolated_only.plot(label='Interpolated Data', color='cyan', linestyle='--')
        mean_forecast.plot(label='Forecast', color='red')
        plt.fill_between(result_df.index, result_df['lower_bound'], result_df['upper_bound'], color='pink', alpha=0.5, label=f'{confidence_interval*100}% Confidence Interval')
        plt.title(f'Forecast for next {steps} steps with Confidence Interval')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
            
        return result_df
