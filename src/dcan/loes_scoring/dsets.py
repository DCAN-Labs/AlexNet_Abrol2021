import csv
import functools
import glob
import logging
import os
from collections import namedtuple
from datetime import datetime

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
