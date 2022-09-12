import pandas as pd
from dataclasses import dataclass

file_path = \
    '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring/9_7 MRI sessions Igor Loes score updated.csv'
partial_loes_scores_df = pd.read_csv(file_path, header=[0, 1])
print(partial_loes_scores_df.head())