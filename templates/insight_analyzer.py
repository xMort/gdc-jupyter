import os
import requests
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
        return None

    def _post_to_server(self, result_json, path):
        url = f"{self.host}/api/v1/actions/workspaces/{self.workspace_id}/cache/{path}/{self.result_id}"
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}',
        }
        # sending get request and saving the response as response object
        r = requests.post(url=url, headers=headers, data=result_json)
        print("Success!")

    def get_df(self):
        return self._fetch_data()


class ClusterAnalyzer(InsightAnalyzer):
    def __init__(
            self,
            result_id: str,
            workspace: str | None = None,
            host_name: str | None = None,
            api_token: str | None = None,
    ):
        super().__init__(
            result_id=result_id,
            workspace=workspace,
            host_name=host_name,
            api_token=api_token
        )

    def _fetch_data(self):
        if self.df is None:
            df = self.gp.data_frames(self.workspace_id).for_exec_result_id(
                self.result_id
            )[0]
            if len(df.index) < 10:
                df = df.T
            self.df = df
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

        data = {"clusters": clusters}
        result_json = json.dumps(data).decode()

        try:
            self._post_to_server(result_json, "clustering")
        except Exception as e:
            print("Could not send the JSON to the server printing data:")
            print(result_json)


class ForecastAnalyzer(InsightAnalyzer):
    def __init__(
            self,
            result_id: str,
            workspace: str | None = None,
            host_name: str | None = None,
            api_token: str | None = None,
    ):
        super().__init__(
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
        try:
            self._post_to_server(result_json, "forecast")
        except Exception as e:
            print("Could not post to the server.")


class AnomalyAnalyzer(InsightAnalyzer):
    def __init__(
            self,
            result_id: str,
            workspace: str | None = None,
            host_name: str | None = None,
            api_token: str | None = None,
    ):
        super().__init__(
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
            if len(tmp.index) < 10:
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
        try:
            self._post_to_server(result_json, "anomalyDetection")
        except Exception as e:
            print("Could not send the JSON to the server printing data:")
            print(e)
