import csv
import os.path
from os import listdir
from os.path import isfile, join

import torch
import torchio as tio


parent_folder = '/home/feczk001/shared/data/loes_scoring/Loes_score/'
nifti_ext = '.nii.gz'
spreadsheet = os.path.join(parent_folder, 'loes_scores.csv')
new_spreadsheet = os.path.join(parent_folder, 'loes_scores_new.csv')

with open(new_spreadsheet, "w") as new_f:
    new_f.write('Sub ID,Sub_Session,Loes score,validated,flair,mprage,swi,flair_size,mprage_size,swi_size\n')
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
                if flair:
                    flair_image = tio.ScalarImage(os.path.join(session_folder, 'flair.nii.gz'))
                    data = flair_image.data
                    data = torch.squeeze(data, dim=0)
                    flair_shape = list(data.shape)
                else:
                    flair_shape = ''
                mprage = 1 if 'mprage' in file_types_in_dir else 0
                if mprage:
                    mprage_image = tio.ScalarImage(os.path.join(session_folder, 'mprage.nii.gz'))
                    data = mprage_image.data
                    data = torch.squeeze(data, dim=0)
                    mprage_shape = list(data.shape)
                else:
                    mprage_shape = ''
                swi = 1 if 'swi' in file_types_in_dir else 0
                if swi:
                    swi_image = tio.ScalarImage(os.path.join(session_folder, 'swi.nii.gz'))
                    data = swi_image.data
                    data = torch.squeeze(data, dim=0)
                    swi_shape = list(data.shape)
                else:
                    swi_shape = ''
                new_f.write(f'{",".join(row)},{flair_shape},{mprage_shape},{swi_shape}\n')
