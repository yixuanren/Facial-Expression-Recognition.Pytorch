import numpy as np

from pdb import set_trace


preds = np.load('preds.npz')['arr_0']
_, counts = np.unique(preds, return_counts=True)

rep_list = np.load('rep_list.npz')['arr_0']

total = 20000
selected = np.zeros(total, dtype=int)

minor = [0, 1, 2, 4, 5]
minor_cum = np.cumsum([0, counts[0], counts[1], counts[2], counts[4], counts[5]])
for i in range(len(minor)):
	indices = np.where(preds==minor[i])[0]
	selected[minor_cum[i]:minor_cum[i+1]] = indices

major = [3, 6]
sub_total = (total - minor_cum[-1]) / 2
for i in range(len(major)):
	indices = np.where(preds==major[i])[0]
	chosen = np.random.choice(indices, size=sub_total, replace=False)
	selected[-sub_total*(i+1):total-sub_total*i] = chosen

set_trace()

np.savez_compressed('selected', selected)
print('selected.npz saved')