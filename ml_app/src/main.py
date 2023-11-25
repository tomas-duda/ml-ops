import pandas as pd


class TitanicDataCleaner:

    path: str
    def __init__(self, path: str):
        self.path = path

    def clean(self):
        with open(self.path, 'r') as f:
            pd.read_csv(self.path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    TitanicDataCleaner("C:\\Users\\tduda\\Documents\\projects\\Test\\ml-ops\\data\\titanic.csv").clean()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
