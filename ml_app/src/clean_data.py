import argparse
import os

import pandas
import pandas as pd
import numpy as np


class TitanicDataCleaner:
    """
    Wrapper class for methods used to clean Titanic data.
    """

    base_path: str

    def __init__(self, base_path: str):
        """

        :param base_path: Path to the CSV with titanic data.
        """
        self.base_path = base_path

    def clean(self):
        df = self._read_data(self.base_path + os.sep + "raw" + os.sep + "titanic.csv")
        df = self._clean_data(df)
        self._write_data(self.base_path + os.sep + "cleaned" + os.sep + "titanic.csv", df)

    def _read_data(self, path: str):
        data = pd.read_csv(path)
        return data

    def _clean_data(self, data: pd.DataFrame) -> pandas.DataFrame:
        """
        Runs cleaning steps for Titanic data.

        :return: Dataframe with cleaned data.
        """
        # Read data form provided path
        df_t1 = data[['passenger_id', 'ticket', 'pclass', 'fare', 'sex', 'age', 'cabin', 'embarked']]

        # Clean rows with NA values
        df_t1 = df_t1[df_t1['fare'].notna() & (df_t1['fare'] > 0) & (df_t1['embarked'].notna())]

        # Making new dataset of tickets
        # User function
        def rate_males(s):
            return np.mean(np.where(s == 'male', 1, 0))

        # Base table
        df_t2_base = df_t1[['ticket', 'pclass', 'fare']].drop_duplicates()
        df_t2_base = df_t2_base.set_index('ticket')  # setting 'ticket' column as key

        # Multiple embarkment solution
        df_t2_emb = df_t1.groupby('ticket').agg({'embarked': 'max'})
        # no need to set index - groupby + agg sets index by default

        # Some chosen features
        df_t2_feat = df_t1.groupby('ticket').agg({'ticket': 'count', 'sex': [rate_males],
                                                  'age': ['min', 'max', np.mean, 'count'], 'cabin': 'nunique'})
        # Column names update
        df_t2_feat.columns = ['pass_cnt', 'rate_males', 'age_min', 'age_max', 'age_mean', 'age_valid_cnt',
                              'cabin_cnt']

        # Sex of the oldest person for the ticket
        df_t2_feat_sex_oldest = df_t1.sort_values(by=['ticket', 'age'], ascending=[True, False]) \
            .drop_duplicates('ticket')[['ticket', 'sex']]
        df_t2_feat_sex_oldest = df_t2_feat_sex_oldest.set_index('ticket')  # setting 'ticket' column as key
        df_t2_feat_sex_oldest.columns = ['sex_oldest']

        # Joining tables together
        df_t2 = df_t2_base.join(df_t2_emb)  # join is by default LEFT and index<->index
        df_t2 = df_t2.join(df_t2_feat)
        df_t2 = df_t2.join(df_t2_feat_sex_oldest)

        return df_t2

    def _write_data(self, path: str, data: pd.DataFrame):
        """
        Path where the cleaned data will be written.

        :param path: Local path.
        """
        data.to_csv(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Titanic Data Cleaner",
                                     description="Cleans Titanic data")
    parser.add_argument('-i', '--input',
                        default="C:\\Users\\tduda\\Documents\\projects\\Test\\ml-ops\\data\\")
    args = parser.parse_args()
    TitanicDataCleaner(args.input).clean()
