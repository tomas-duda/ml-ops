import argparse
import os

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import statsmodels.formula.api as smf


class TitanicModel:
    """
    Wrapper class for methods used to fit Titanic model.
    """

    base_path: str

    def __init__(self, base_path: str):
        """

        :param base_path: Path to the CSV with titanic data.
        """
        self.base_path = base_path


    def model(self):
        os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000/"
        df = self._read_data(self.base_path + os.sep + "features" + os.sep + "titanic.csv")
        self._fit_model(df)

    def _read_data(self, path: str):
        data = pd.read_csv(path)
        return data

    def _fit_model(self, data: pd.DataFrame):
        """
        Runs Titanic model training.

        :return: Dataframe with features.
        """
        X = data[['pass_cnt', 'pclass']]
        y = data['fare_log']

        with mlflow.start_run() as run:
            # fit model
            modelA = LinearRegression().fit(X, y)

            # get coefficients
            # print('Intercept: ', modelA.intercept_)
            # print('Beta coefficients: ', modelA.coef_)
            scores = cross_val_score(LinearRegression(), X, y, cv=4)
            print('R2 by cval: ', scores)

            mlflow.sklearn.log_model(modelA, "model")
            mlflow.log_metric("r2_mean", scores.mean())
            mlflow.log_param("schema", list(X.columns))
            mlflow.log_param("model_class", type(modelA))

            modelB = smf.ols("fare_log ~ pass_cnt + pclass", data=data).fit()
            ols_summary = modelB.summary()
            print(ols_summary)
            mlflow.log_text(str(ols_summary), "ols_summary.txt")  # detailed information of model and coefficients


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Titanic Model",
                                     description="Trains Titanic Model")
    parser.add_argument('-i', '--input',
                        default="C:\\Users\\tduda\\Documents\\projects\\Test\\ml-ops\\data\\")
    args = parser.parse_args()
    TitanicModel(args.input).model()
