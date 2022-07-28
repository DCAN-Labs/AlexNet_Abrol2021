import csv
import os

original_file = '/home/feczk001/shared/data/loes_scoring/Loes_score_validated/validated_loes_scores.csv'
loews_score_folder = '/home/feczk001/shared/data/loes_scoring/Loes_score/'
with open(original_file, "r") as f:
    for row in list(csv.reader(f))[1:]:
        ald_code = row[0].strip()
        subject_folder = os.path.join(loews_score_folder, f'sub-{ald_code}')
        session_folders = [directory[0] for directory in os.walk(subject_folder)]
        session_folders.remove(subject_folder)
        sessions = [s[s.rfind('-') + 1:] for s in session_folders]
        loes_score = row[2].lstrip()
        for session in sessions:
            print(f'{ald_code},{ald_code}_{session},{loes_score},1')
