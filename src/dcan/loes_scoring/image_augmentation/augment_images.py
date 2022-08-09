import os.path
import csv
from pathlib import Path

import pandas as pd
import torchio as tio


csv_folder = 'data/loes_scoring'
df = pd.read_csv(os.path.join(csv_folder, 'ALD-google_sheet-Jul272022-Loes_scores-Igor.csv'))
loes_scoring_folder = '/home/feczk001/shared/data/loes_scoring'
with open(os.path.join(csv_folder, 'augmented_files.csv'), 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['_Sub ID', 'Sub_Session', 'Loes score', 'augmentation_type', 'augmentation_index'])
    df = df[df["Loes score"] > 0]
    value_counts = df['Loes score'].value_counts()
    df = df.reset_index()
    for index, row in df.iterrows():
        loes_score = row['Loes score']
        if loes_score > 0:
            subject_session_id = row['Sub_Session']
            subject_id, session_id = subject_session_id.split('_')
            folder = os.path.join(loes_scoring_folder, f'Loes_score/sub-{subject_id}/ses-{session_id}')
            augmented_folder = \
                os.path.join(loes_scoring_folder, f'Loes_score_augmented/sub-{subject_id}/ses-{session_id}')
            path = os.path.join(folder, 'mprage.nii.gz')
            image = tio.ScalarImage(path)
            for i in range(10 - value_counts[loes_score]):
                Path(augmented_folder).mkdir(parents=True, exist_ok=True)
                transform = tio.RandomElasticDeformation(
                    num_control_points=(7, 7, 7),  # or just 7
                    locked_borders=2,
                )
                transformed = transform(image)
                new_path = os.path.join(augmented_folder, f'mprage_{i}.nii.gz')
                transformed.save(new_path)
                csv_writer.writerow([subject_id, subject_session_id, loes_score, 'RandomElasticDeformation', i])
