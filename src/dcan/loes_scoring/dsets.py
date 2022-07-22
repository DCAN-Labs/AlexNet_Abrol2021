import csv
import functools
import glob
import logging
import os
import copy
from collections import namedtuple
from datetime import datetime

import torch
from torch.utils.data import Dataset
import random

import torchio as tio

from util.disk import getCache

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('dcan_loes_score')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'loes_score_pre_transplant_int, ald_code_ses_uid, ald_code_str, mrn_int, ses_date',
)


def get_subject(p):
    return os.path.split(os.path.split(os.path.split(p)[0])[0])[1][4:]


def get_session(p):
    return os.path.split(os.path.split(p)[0])[1][4:]


def get_uid(p):
    return f'{get_subject(p)}_{get_session(p)}'


@functools.lru_cache(1)
def get_candidate_info_list(require_on_disk_bool=True):
    # We construct a set with all ald_code_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    loes_score_validated_path = '/home/feczk001/shared/data/loes_scoring/Loes_score_validated/sub-*/ses-*/'
    nifti_ext = '.nii.gz'
    dmri_12dir_mri_list = glob.glob(f'{loes_score_validated_path}dmri_12dir{nifti_ext}')
    dmri_12dir_present_on_disk_set = {get_uid(p) for p in dmri_12dir_mri_list}
    mprage_mri_list = glob.glob(f'{loes_score_validated_path}mprage{nifti_ext}')
    mprage_present_on_disk_set = {get_uid(p) for p in mprage_mri_list}
    present_on_disk_set = dmri_12dir_present_on_disk_set.intersection(mprage_present_on_disk_set)

    candidate_info_list = []
    scores_csv = '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring/validated_loes_scores.csv'
    with open(scores_csv, "r") as f:
        for row in list(csv.reader(f))[1:]:
            ses_str = row[1].strip()
            mrn_str = row[2].strip()
            ald_code_str = row[0]
            ald_code_ses_uid = '_'.join([ald_code_str, ses_str])

            if ald_code_ses_uid not in present_on_disk_set and require_on_disk_bool:
                continue

            mrn_int = int(mrn_str)
            loes_score_pre_transplant_int = int(row[3])
            ses_date = datetime.strptime(ses_str, '%Y%m%d')

            candidate_info_list.append(CandidateInfoTuple(
                loes_score_pre_transplant_int,
                ald_code_ses_uid,
                ald_code_str,
                mrn_int,
                ses_date,
            ))

    candidate_info_list.sort(reverse=True)

    return candidate_info_list


class LoesScoreMRIs:
    def __init__(self, subject_session_uid):
        parts = subject_session_uid.split('_')
        subject = parts[0]
        session = parts[1]
        loes_score_validated_dir = \
            f'/home/feczk001/shared/data/loes_scoring/Loes_score_validated/sub-{subject}/ses-{session}/'
        nifti_ext = '.nii.gz'

        dmri_12dir_path = glob.glob('{}dmri_12dir{}'.format(loes_score_validated_dir, nifti_ext))[0]
        dmri_12dir_image = tio.ScalarImage(dmri_12dir_path)
        self.dmri_12dir_tensor = dmri_12dir_image.data

        mprage_path = glob.glob('{}mprage{}'.format(loes_score_validated_dir, nifti_ext))[0]
        mprage_image = tio.ScalarImage(mprage_path)
        log.info(f'Initial shape of image: {mprage_image.shape}')
        transform = tio.CropOrPad(
            (208, 300, 320),
        )
        transformed_mprage_image = transform(mprage_image)
        log.info(f'Shape of transformed image: {transformed_mprage_image.shape}')
        self.mprage_image_tensor = transformed_mprage_image.data

        self.subject_session_uid = subject_session_uid

    def get_raw_candidate(self):
        return self.dmri_12dir_tensor, self.mprage_image_tensor


@functools.lru_cache(1, typed=True)
def get_loes_score_mris(subject_session_uid):
    return LoesScoreMRIs(subject_session_uid)


@raw_cache.memoize(typed=True)
def get_mri_raw_candidate(subject_session_uid):
    loes_score_mris = get_loes_score_mris(subject_session_uid)
    dmri_12dir_tensor, mprage_image_tensor = loes_score_mris.get_raw_candidate()

    return dmri_12dir_tensor, mprage_image_tensor


class LoesScoreDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 is_val_set_bool=None,
                 subject=None,
                 sortby_str='random',
                 ):
        self.candidateInfo_list = copy.copy(get_candidate_info_list())

        if subject:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.ald_code_str == subject
            ]

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'subject':
            self.candidateInfo_list = sorted(self.candidateInfo_list, key=CandidateInfoTuple.ald_code_str.fget)
        elif sortby_str == 'loes_score':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if is_val_set_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidate_info = self.candidateInfo_list[ndx]
        subject_session_uid = candidate_info.ald_code_ses_uid
        # TODO Possibly handle other file types such as diffusion-weighted sequences
        _, mprage_image_tensor = get_mri_raw_candidate(subject_session_uid)
        loes_score = candidate_info.loes_score_pre_transplant_int
        loes_score_t = torch.tensor(loes_score, dtype=torch.long)

        return mprage_image_tensor, loes_score_t
