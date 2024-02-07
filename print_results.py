from glob import glob

import pandas as pd

from configs import DATASET_CONFIGS


COLS = [
	'lev_ratio', 'min_k', 'quiz_score', 'kappa', 'neighbor_loss_delta', 'guided_rouge', 'general_rouge', 'p_value', 'ppl', 'lower_ratio', 'zlib_lratio'
]


if __name__ == '__main__':
	results_fn = list(glob('results/*.csv'))
	for dataset in sorted(DATASET_CONFIGS.keys()):
		fns = [x for x in results_fn if dataset in x]
		print(dataset + f'({len(fns)})')
		for fn in fns:
			df = pd.read_csv(fn)
			strs = []
			for col in COLS:
				if col not in df:
					print(f'Missing {col} in {fn}')
					strs.append('-')
				else:
					strs.append(str(round(df[col].dropna().mean(), 3)))
			print(fn)
			print('\t' + ' '.join(strs))
			print('\n--------\n')
