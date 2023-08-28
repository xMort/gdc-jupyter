import os
from dotenv import load_dotenv
from gooddata_pandas import GoodPandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from numpy import where, unique
import numpy as np


class InsightEnvironment:
    def __init__(self, insight_id: str, workspace: str | None = None):
        load_dotenv()
        self.host = os.getenv("HOST")
        self.token = os.getenv("TOKEN")
        if workspace is None:
            self.workspace_id = os.getenv("WORKSPACE")
        else:
            self.workspace_id = workspace
        self.insight_id = insight_id


def describe_data(env: InsightEnvironment):
    gp = GoodPandas(host=env.host, token=env.token)
    df = gp.data_frames(env.workspace_id).for_insight(env.insight_id)
    df.describe()

def show_points(env: InsightEnvironment):
    gp = GoodPandas(host=env.host, token=env.token)
    df = gp.data_frames(env.workspace_id).for_insight(env.insight_id)
    dim_y = df.columns.values[0]
    dim_x = df.columns.values[1]
    plt.scatter(df[dim_x], df[dim_y])
    plt.show()
    
def cluster(env: InsightEnvironment, custer_count: int = 3):
    gp = GoodPandas(host=env.host, token=env.token)
    df = gp.data_frames(env.workspace_id).for_insight(env.insight_id)
    dim_y = df.columns.values[0]
    dim_x = df.columns.values[1]
    X = np.column_stack((df[dim_x],df[dim_y]))

    model = AgglomerativeClustering(n_clusters=custer_count)

    yhat = model.fit_predict(X)

    clusters = unique(yhat)
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        # show the plot
    plt.show()


def show_train_data(env: InsightEnvironment):
    gp = GoodPandas(host=env.host, token=env.token)
    df = gp.data_frames(env.workspace_id).for_insight(env.insight_id)
    df = df.asfreq('H')

    # number of steps we will try to predict
    steps = round(len(df) * 0.1)
    data_train = df[:-steps]
    data_test = df[-steps:]
    dim_y = df.columns.values[0]

    print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    fig, ax = plt.subplots(figsize=(7, 2.5))
    data_train[dim_y].plot(ax=ax, label='train')
    data_test[dim_y].plot(ax=ax, label='test')
    ax.legend()


def predict(env: InsightEnvironment):
    gp = GoodPandas(host=env.host, token=env.token)
    df = gp.data_frames(env.workspace_id).for_insight(env.insight_id)
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
    data_interpolate = data_train
    data_interpolate[dim_y] = data_interpolate[dim_y].interpolate(method='linear')

    forecaster.fit(y=data_interpolate[dim_y])

    
    predictions = forecaster.predict(steps=steps)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    data_interpolate[dim_y].plot(ax=ax, label='train')
    data_test[dim_y].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend()

    error_mse = mean_squared_error(
        y_true = data_test[dim_y],
        y_pred = predictions
    )

    print(f"Test error (mse): {error_mse}")