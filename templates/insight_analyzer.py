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
from statsmodels.tsa.statespace.sarimax import SARIMAX
import simplejson as json
from pandas import DataFrame


class InsightAnalyzer:

    def __init__(self, insight_id: str, workspace: str | None = None):
        load_dotenv()
        self.host = os.getenv("HOST")
        self.token = os.getenv("TOKEN")
        self.workspace_id = os.getenv("WORKSPACE") if workspace is None else workspace
        self.insight_id = insight_id
        self.gp = GoodPandas(host=self.host, token=self.token)
        self.df = None

    def _fetch_data(self):
        if self.df is None:
            self.df = self.gp.data_frames(self.workspace_id).for_insight(self.insight_id)
        return self.df

    def get_df(self):
        return self._fetch_data()
    def describe_data(self):
        df = self._fetch_data()
        return df.describe()

    def show_data(self):
        raise Exception("Base has no data")


class ClusterAnalyzer(InsightAnalyzer):

    def __init__(self, insight_id: str, workspace: str | None = None):
        InsightAnalyzer.__init__(self, insight_id = insight_id, workspace = workspace)

    def cluster(self, cluster_count: int = 3, threshold: int = 0.05, new_df: DataFrame = None):
        df = self._fetch_data()

        if new_df is not None:
            x = np.column_stack((new_df[df.columns[1]], new_df[df.columns[0]]))
        else:
            x = np.column_stack((df[df.columns[1]], df[df.columns[0]]))

        model = Birch(threshold=threshold, n_clusters=cluster_count)
        yhat = model.fit_predict(x)

        clusters = []
        for cluster in np.unique(yhat):
            cluster_data = [
                [x, y] for x, y in zip(
                    df.values[yhat == cluster, 0],
                    df.values[yhat == cluster, 1])
            ]
            clusters.append(cluster_data)

        for cluster in np.unique(yhat):
            plt.scatter(df.values[yhat == cluster, 1], df.values[yhat == cluster, 0])
        plt.show()

        return json.dumps({"clusters": clusters})

    def show_data(self):
        df = self._fetch_data()
        plt.scatter(
            df[df.columns[1]],
            df[df.columns[0]]
        )
        plt.show()


class PredictionAnalyzer(InsightAnalyzer):

    def __init__(self, insight_id: str, workspace: str | None = None):
        InsightAnalyzer.__init__(self, insight_id = insight_id, workspace = workspace)

    @staticmethod
    def _prepare_train_test_data(df):
        steps = round(len(df) * 0.1)
        data_train = df[:-steps]
        data_test = df[-steps:]
        return data_train, data_test

    def predict(self, steps: int = 50, conf_interval: int = 90):
        df = self._fetch_data()
        df = df.asfreq("H")

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
        data_interpolate = df.copy()
        data_interpolate.loc[:, dim_y] = data_interpolate.loc[:, dim_y].interpolate(
            method="linear"
        )

        forecaster.fit(y=data_interpolate[dim_y])

        predictions = forecaster.predict_interval(
            steps=steps,
            interval=[(100-conf_interval)/2, 100-((100-conf_interval)/2)]
        )

        # Calculate lower and upper bounds
        z_score = np.abs(np.percentile(predictions, (100 ) / 2))
        lower_bound = predictions - z_score * np.std(predictions)
        upper_bound = predictions + z_score * np.std(predictions)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        df[dim_y].plot(ax=ax, label="real")
        predictions["pred"].plot(ax=ax, label="predictions")
        plt.fill_between(
            predictions.index,
            predictions["lower_bound"],
            predictions["upper_bound"],
            color="red",
            alpha=0.7,
            label=f"{conf_interval}% Confidence Interval",
        )
        ax.legend()

        interpolated_data = df[dim_y].interpolate(metho="linear")

        # Create the result dictionary
        data = (
            pd.merge(
                predictions,
                interpolated_data.rename("origin").to_frame(),
                left_index=True,
                right_index=True,
                how="outer"
            )
        )
        result_dict = {
            "attribute": data.index.strftime('%Y-%m-%d %H:%M:%S').to_list(),
            "origin": data['origin'].tolist(),
            "predictions": data[['lower_bound', 'pred', 'upper_bound']].values.tolist()
        }
        return json.dumps(result_dict, ignore_nan=True)

    def show_data(self):
        df = self._fetch_data().asfreq("H")
        data_train, data_test = self._prepare_train_test_data(df)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        data_train[df.columns[0]].plot(ax=ax, label="train")
        data_test[df.columns[0]].plot(ax=ax, label="test")
        ax.legend()


class AnomalyAnalyzer(InsightAnalyzer):

    def __init__(self, insight_id: str, workspace: str | None = None):
        InsightAnalyzer.__init__(self, insight_id = insight_id, workspace = workspace)

    def show_data(self):
        df = self._fetch_data().asfreq("H")

        fig, ax = plt.subplots(figsize=(7, 2.5))
        df[df.columns[0]].plot(ax=ax, label="train")
        ax.legend()

    def anomalies(self):
        df = self._fetch_data().asfreq("H")
        data = df[df.columns[0]].interpolate(method="linear")
        validate_series(data)

        seasonal_ad = PersistAD(c=3, side="both")
        anomalies = seasonal_ad.fit_detect(data)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        data.plot(ax=ax, label="train")
        anomalies = anomalies.fillna(False)

        # Filter the data using the anomalies binary mask to get the anomaly values.
        anomaly_values = data[anomalies]

        # Use scatter to plot the anomalies as points.
        ax.scatter(anomaly_values.index, anomaly_values, color="red", label="anomalies")

        ax.legend()
