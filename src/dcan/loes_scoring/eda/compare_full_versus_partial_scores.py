import os.path
import pandas as pd
import numpy as np


from dcan.loes_scoring.data.partial_loes_scores import get_partial_loes_scores

loes_scoring_folder = '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring'
partial_scores_file_path = os.path.join(loes_scoring_folder, '9_7 MRI sessions Igor Loes score updated.csv')
partial_loes_scores = get_partial_loes_scores(partial_scores_file_path)
full_score_path = os.path.join(loes_scoring_folder, 'ALD-google_sheet-Jul272022-Loes_scores-Igor.csv')
full_scores_df = pd.read_csv(full_score_path)
for ind in full_scores_df.index:
    loes_score_field_name = 'Loes score'
    score = full_scores_df[loes_score_field_name][ind]
    if not np.isnan(score):
        continue
    sub_id = full_scores_df['_Sub ID'][ind]
    if sub_id in partial_loes_scores:
        sub_session = full_scores_df['Sub_Session'][ind]
        if sub_session in partial_loes_scores[sub_id]:
            partial_score_score = partial_loes_scores[sub_id][sub_session]
            if partial_score_score.loes_score != 0:
                full_scores_df.at[ind, loes_score_field_name] = partial_score_score.loes_score
updated_full_score_path = os.path.join(loes_scoring_folder, 'ALD-google_sheet-Jul272022-Loes_scores-Igor_updated.csv')
full_scores_df.to_csv(updated_full_score_path, index=False)
