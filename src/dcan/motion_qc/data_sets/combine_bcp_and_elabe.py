import os.path

import pandas as pd
import numpy as np


def get_file_path(row):
    src_folder = '/home/elisonj/shared/BCP/raw/BIDS_output'
    file_path = os.path.join(src_folder, row['path'])

    return file_path


data_folder = '/home/miran045/reine097/projects/AlexNet_Abrol2021/data'
bcp_df = pd.read_csv(os.path.join(data_folder, 'BCP/qc_with_paths.csv'))
bcp_df['image_path'] = bcp_df.apply(lambda row: get_file_path(row), axis=1)
bcp_df.rename(columns = {'motionQCscore': 'rating'}, inplace = True)
bcp_df['data_set'] = 'BCP'
bcp_df = bcp_df[['image_path', 'rating', 'data_set']]

elabe_df = pd.read_csv(os.path.join(data_folder, 'eLabe/qc_img_paths.csv'))
elabe_df['data_set'] = 'eLabe'
elabe_df = elabe_df[['image_path', 'rating', 'data_set']]

df = pd.concat([bcp_df, elabe_df], ignore_index=True, axis=0)
df = df.reset_index(drop=True)

fractions = np.array([0.9, 0.1])
df = df.sample(frac=1)
train, test = np.array_split(
    df, (fractions[:-1].cumsum() * len(df)).astype(int))
train.to_csv(os.path.join(data_folder, 'BCP_and_eLabe', 'bcp_and_elabe_qc_train.csv'))
test.to_csv(os.path.join(data_folder, 'BCP_and_eLabe', 'bcp_and_elabe_qc_test.csv'))
