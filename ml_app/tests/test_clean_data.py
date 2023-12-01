from clean_data import TitanicDataCleaner

import os


def test_clean_data():
    tdc = TitanicDataCleaner(os.path.join(os.path.dirname(__file__), f"data"))
    tdc.clean()
    assert os.path.exists(os.path.join(os.path.dirname(__file__), f"data{os.sep}cleaned{os.sep}titanic.csv"))