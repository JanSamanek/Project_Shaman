import numpy as np
from scipy.spatial import distance
import math

coords = [(0, 0),
          (-7, 2),
          (4, 2),
          (5, 3)]

coords1 = [(1, 0),
           (-9, -1),
           (8,1),
           (-3, 4)]

D = distance.cdist(np.array(coords), np.array(coords1))

rows_min_i = D.argmin(axis=1)
columns_min_i = D.argmin(axis=0)


row_len = D.shape[0]
column_len = D.shape[1]

used_rows = set()
used_columns = set()

print(D)
print(rows_min_i)
print(columns_min_i)

for row in range(0, row_len):
    row_min_i = rows_min_i[row]
    # column where the minimum of the row is
    column = row_min_i
    column_min_i = columns_min_i[row]
    if row_min_i == column_min_i:
        # link points together
        used_columns.add(column)
        used_rows.add(row)


print(D)
print(used_rows)
print(used_columns)


