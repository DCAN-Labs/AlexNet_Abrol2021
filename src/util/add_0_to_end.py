import os.path

loes_scoring_dir = '/home/feczk001/shared/data/loes_scoring'
with open(os.path.join(loes_scoring_dir, 'Loes_score/Loes_scores_new.csv'), 'w') as new_f:
    old_file_path = \
        os.path.join(loes_scoring_dir, 'Loes_score_unvalidated', 'ALD-google_sheet-Jul272022-Loes_scores-Igor.csv')
    with open(old_file_path, 'r') as old_f:
        lines = old_f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if i == 0:
                new_f.write('Sub ID,Sub_Session,Loes score,validated\n')
            else:
                new_f.write(f'{line},0\n')
