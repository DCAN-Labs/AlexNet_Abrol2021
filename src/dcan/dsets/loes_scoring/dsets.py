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
    '''ald_code_mrn_uid, ald_code_uid, mrn_int, ses_date, loes_score_pre_transplant_int, dmri_12dir_path_str, '''
    '''mprage_path_str'''
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
    dmri_12dir_mri_list = glob.glob(
        '/home/feczk001/shared/data/loes_scoring/Loes_score_validated/sub-*/ses-*/dmri_12dir.nii.gz')
    dmri_12dir_present_on_disk_set = {get_uid(p) for p in dmri_12dir_mri_list}
    present_on_disk_set = {}

    candidate_info_list = []
    with open('data/loes_scoring/validated_loes_scores.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            ses_str = row[1].strip()
            mrn_str = row[2].strip()
            ald_code_uid = row[0]
            ald_code_ses_mrn_uid = '_'.join([ald_code_uid, ses_str, mrn_str])

            if ald_code_ses_mrn_uid not in present_on_disk_set and require_on_disk_bool:
                continue

            mrn_int = int(mrn_str)
            loes_score_pre_transplant_int = int(row[3])
            base_dir = '/home/feczk001/shared/data/loes_scoring/Loes_score_validated'
            session_dir = os.path.join(base_dir, f'sub-{ald_code_uid}', f'ses-{ses_str}')
            dmri_12dir_path_str = os.path.join(session_dir, 'dmri_12dir.nii.gz')
            mprage_path_str = os.path.join(session_dir, 'mprage.nii.gz')
            ses_date = datetime.strptime(ses_str, '%Y%m%d')

            candidate_info_list.append(CandidateInfoTuple(
                ald_code_ses_mrn_uid,
                ald_code_uid,
                mrn_int,
                ses_date,
                loes_score_pre_transplant_int,
                dmri_12dir_path_str,
                mprage_path_str,
            ))

    candidate_info_list.sort(reverse=True)

    return candidate_info_list
