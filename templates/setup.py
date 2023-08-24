import os
from dotenv import load_dotenv
from gooddata_sdk import GoodDataSdk
from gooddata_pandas import GoodPandas
from pandas import DataFrame


class InsightEnvironment:
    def __init__(self, insight_id: str):
        load_dotenv()
        self.host = os.getenv("HOST")
        self.token = os.getenv("TOKEN")
        self.workspace_id = os.getenv("WORKSPACE")
        self.insight_id = insight_id
