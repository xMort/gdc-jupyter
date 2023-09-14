import os
from dotenv import load_dotenv
from gooddata_pandas import GoodPandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from adtk.data import validate_series
from adtk.detector import PersistAD
import orjson as json
from pandas import DataFrame
import urllib
from sklearn.preprocessing import MinMaxScaler


class InsightAnalyzer:
    def __init__(
        self,
        result_id: str,
        workspace: str | None = None,
        host_name: str | None = None,
        api_token: str | None = None,
    ):
        load_dotenv()
        self.host = os.getenv("HOST") if host_name is None else host_name
        self.token = os.getenv("TOKEN") if api_token is None else api_token
        self.workspace_id = os.getenv("WORKSPACE") if workspace is None else workspace
        self.result_id = result_id
        self.gp = GoodPandas(host=self.host, token=self.token)
        self.df = None

    def _fetch_data(self):
        if self.df is None:
            df = self.gp.data_frames(self.workspace_id).for_exec_result_id(
                self.result_id
            )[0]
            self.df = DataFrame(df.T.iloc[:, 0])
        return self.df

    def get_df(self):
        return self._fetch_data()

    def describe_data(self):
        df = self._fetch_data()
        return df.describe()

    def show_data(self):
        raise Exception("Base has no data")


class ClusterAnalyzer(InsightAnalyzer):
    def __init__(
            self,
            result_id: str,
            workspace: str | None = None,
            host_name: str | None = None,
            api_token: str | None = None,
    ):
        InsightAnalyzer.__init__(
            self,
            result_id=result_id,
            workspace=workspace,
            host_name=host_name,
            api_token=api_token
        )

    def cluster(self, cluster_count: int = 3, threshold: int = 0.05):
        df = self._fetch_data()

        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_df = scaler.fit_transform(df)
        normalized_df = pd.DataFrame(normalized_df, columns=df.columns)
        x = np.column_stack(
            (
                normalized_df[normalized_df.columns[0]],
                normalized_df[normalized_df.columns[1]],
            )
        )

        model = Birch(threshold=threshold, n_clusters=cluster_count)
        yhat = model.fit_predict(x)

        clusters = []
        for cluster in np.unique(yhat):
            cluster_data = [
                [x, y]
                for x, y in zip(
                    df.values[yhat == cluster, 0], df.values[yhat == cluster, 1]
                )
            ]
            clusters.append(cluster_data)

        for cluster in np.unique(yhat):
            plt.scatter(df.values[yhat == cluster, 0], df.values[yhat == cluster, 1])
        plt.show()

        return json.dumps({"clusters": clusters})

    def _fetch_data(self):
        if self.df is None:
            df = self.gp.data_frames(self.workspace_id).for_exec_result_id(
                self.result_id
            )[0]
            if len(df.index) > 10:
                df = df.T
            self.df = df.T
            return self.df
        return self.df

    def push_to_server(self, yhat):
        df = self._fetch_data()
        clusters = []
        for cluster in np.unique(yhat):
            cluster_indices = df.index[yhat == cluster].tolist()
            cluster_data = [
                [index[0], int(x), int(y)]
                for index, x, y in zip(
                    cluster_indices, df.values[yhat == cluster, 0], df.values[yhat == cluster, 1]
                )
            ]
            clusters.append(cluster_data)

        result_json = json.dumps({"clusters": clusters}).decode()

        if self.result_id is not None:
            try:
                req = urllib.request.Request(
                    "http://localhost:8080/set?id=" + self.result_id
                )
                req.add_header("Content-Type", "application/json; charset=utf-8")
                json_data_bytes = result_json.encode("utf-8")
                urllib.request.urlopen(req, json_data_bytes)
                print("Success!")
            except Exception as e:
                print("Could not send the JSON to the server")
        else:
            print(result_json)


    def show_data(self):
        df = self._fetch_data()
        plt.scatter(df[df.columns[0]], df[df.columns[1]])
        plt.show()


class ForecastAnalyzer:
    def __init__(
            self,
            result_id: str,
            workspace: str | None = None,
            host_name: str | None = None,
            api_token: str | None = None,
    ):
        InsightAnalyzer.__init__(
            self,
            result_id=result_id,
            workspace=workspace,
            host_name=host_name,
            api_token=api_token
        )

    def _fetch_data(self):
        if self.df is None:
            tmp = self.gp.data_frames(self.workspace_id).for_exec_result_id(
                self.result_id
            )[0]
            tmp = tmp.T
            tmp.index = pd.to_datetime(
                ["Q".join(x[0].split("-")) for x in tmp.index]
            ).to_period("Q")
            tmp.columns = [x[0] for x in tmp.columns]
            self.df = tmp
        return self.df

    def get_df(self):
        return self._fetch_data()

    def push_to_server(self, prediction: DataFrame, data: DataFrame):
        data.rename(columns={data.columns[0]: "origin"}, inplace=True)

        data = pd.merge(
            prediction, data, left_index=True, right_index=True, how="outer"
        )

        result_dict = {
            "attribute": data.index.strftime("%Y-%m-%d %H:%M:%S").to_list(),
            "origin": data["origin"].tolist(),
            "predictions": data[
                ["lower_bound", "forecast", "upper_bound"]
            ].values.tolist(),
        }

        result_json = json.dumps(result_dict).decode()
        if self.result_id is not None:
            try:
                req = urllib.request.Request(
                    "http://localhost:8080/set?id=" + self.result_id
                )
                req.add_header("Content-Type", "application/json; charset=utf-8")
                json_data_bytes = result_json.encode("utf-8")
                urllib.request.urlopen(req, json_data_bytes)
                print("Success!")
            except Exception as e:
                print("Could not send the JSON to the server")
        else:
            print(result_json)


class AnomalyAnalyzer(InsightAnalyzer):
    def __init__(
            self,
            result_id: str,
            workspace: str | None = None,
            host_name: str | None = None,
            api_token: str | None = None,
    ):
        InsightAnalyzer.__init__(
            self,
            result_id=result_id,
            workspace=workspace,
            host_name=host_name,
            api_token=api_token
        )

    def _fetch_data(self):
        if self.df is None:
            tmp = self.gp.data_frames(self.workspace_id).for_exec_result_id(
                self.result_id
            )[0]
            if len(tmp.index) <   10:
                tmp = tmp.T
            tmp.index = pd.to_datetime([''.join(i) for i in tmp.index])
            tmp.index = pd.to_datetime(tmp.index)
            self.df = tmp
        return self.df

    def push_to_server(self, anomalies):
        df = self._fetch_data()
        result_dict = {
            "attribute": df.index.strftime("%Y-%m-%d %H:%M:%S").to_list(),
            "values": df[df.columns[0]].tolist(),
            "anomalyFlag": anomalies[anomalies.columns[0]].values.tolist()
        }

        result_json = json.dumps(result_dict).decode()
        if self.result_id is not None:
            print(result_json)
            try:
                req = urllib.request.Request(
                    "http://localhost:8080/set?id=" + self.result_id
                )
                req.add_header("Content-Type", "application/json; charset=utf-8")
                json_data_bytes = result_json.encode("utf-8")
                urllib.request.urlopen(req, json_data_bytes)
                print("Success!")
            except Exception as e:
                print("Could not send the JSON to the server")
        else:
            print(result_json)

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
