from glob import glob

import pandas as pd

if __name__ == '__main__':
	results_fn = list(glob('results/*.csv'))
	for fn in results_fn:
		print(fn)
		print(pd.read_csv(fn).select_dtypes('number').mean())
		print('\n' + '-' * 50 + '\n')
