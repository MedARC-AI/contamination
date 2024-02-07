from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def get_p_value(df, num_tests=10000):
    return ttest_ind(df['guided_rouge'], df['general_rouge'], permutations=num_tests, alternative='greater')[1]
	

if __name__ == '__main__':
    results_fn = list(glob('results/*.csv'))
    for fn in results_fn:
        print(fn)
        df = pd.read_csv(fn)
        p_value = get_p_value(df)
        if p_value < 0.05:
            print(fn + ' -> ' + str(p_value))
