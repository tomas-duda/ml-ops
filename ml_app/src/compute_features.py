import argparse
import os

import pandas
import pandas as pd
import numpy as np


class TitanicFeatures:
    """
    Wrapper class for methods used to compute Titanic features.
    """

    base_path: str

    def __init__(self, base_path: str):
        """

        :param base_path: Path to the CSV with titanic data.
        """
        self.base_path = base_path


    def features(self):
        df = self._read_data(self.base_path + os.sep + "cleaned" + os.sep + "titanic.csv")
        df = self._compute_features(df)
        self._write_data(self.base_path + os.sep + "features" + os.sep + "titanic.csv", df)

    def _read_data(self, path: str):
        data = pd.read_csv(path)
        return data

    def _compute_features(self, data: pd.DataFrame) -> pandas.DataFrame:
        """
        Runs Titanic features computation.

        :return: Dataframe with features.
        """
        # mathematical transformations
        data['fare_log'] = np.log10(data['fare'])
        # we use log10 for better interpretation, but simple log is ok, too.
        data['fare_per_pass'] = data['fare'] / data['pass_cnt']

        # binning, making categories and flags
        ### pass_cnt
        data['pass_cnt_cat'] = pd.cut(data['pass_cnt'], [0, 1, 2, 3, 1000], labels=['1', '2', '3', '4+'])

        ### age_mean
        data['age_mean_cat'] = pd.cut(data['age_mean'], [0, 15, 20, 25, 30, 40, 1000],
                                       labels=['15-', '15-20', '20-25', '25-30', '30-40', '40+'])

        ### cabin_cnt (same approach as pass_cnt)
        data['cabin_cnt_cat'] = pd.cut(data['cabin_cnt'], [0, 1, 2, 1000], right=False, labels=['none', '1', '2+'])

        # flags
        data['flag_child'] = (data['age_min'] < 15)
        data['flag_baby'] = (data['age_min'] < 3)

        return data

    def _write_data(self, path: str, data: pd.DataFrame):
        """
        Path where the cleaned data will be written.

        :param path: Local path.
        """
        data.to_csv(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Titanic Data Features",
                                     description="Computes Titanic Features")
    parser.add_argument('-i', '--input',
                        default="C:\\Users\\tduda\\Documents\\projects\\Test\\ml-ops\\data\\")
    args = parser.parse_args()
    TitanicFeatures(args.input).features()
