import pandas as pd
from sqlalchemy import create_engine

HOST = "localhost"
PORT = 3306
USER = "root"
PASSWORD = "root"
DATABASE = "transfermarkt"


class MySQLConnector:
    def __init__(self, host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE):
        self.engine = create_engine(
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        )

    def extract_data(self, query):
        return pd.read_sql(query, self.engine)
