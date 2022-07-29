import csv
import os.path
from os import listdir
from os.path import isfile, join

parent_folder = '/home/feczk001/shared/data/loes_scoring/Loes_score/'
nifti_ext = '.nii.gz'
spreadsheet = os.path.join(parent_folder, 'Loes_scores.csv')
new_spreadsheet = os.path.join(parent_folder, 'loes_scores_new.csv')

with open(new_spreadsheet, "w") as new_f:
    new_f.write('Sub ID,Sub_Session,Loes score,validated,flair,mprage,swi\n')
    with open(spreadsheet, "r") as old_f:
        for row in list(csv.reader(old_f))[1:]:
            sub_session = row[1]
            subject, session = sub_session.split('_')
            session_folder = os.path.join(parent_folder, f'sub-{subject}', f'ses-{session}')
            if os.path.exists(session_folder):
                nifti_files = \
                    [f for f in listdir(session_folder) if isfile(join(session_folder, f)) and f.endswith(nifti_ext)]
                file_types_in_dir = set([f[:-len(nifti_ext)] for f in nifti_files])
                flair = 1 if 'flair' in file_types_in_dir else 0
                mprage = 1 if 'mprage' in file_types_in_dir else 0
                swi = 1 if 'swi' in file_types_in_dir else 0
                new_f.write(f'{",".join(row)},{flair},{mprage},{swi}\n')
