import csv
import functools
import logging
from collections import namedtuple
from operator import attrgetter

import numpy as np
import torchio as tio

from util.disk import getCache

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('dcan_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'smriPath_str, rating_int'
)


@functools.lru_cache(1)
def get_candidate_info_list(qc_with_paths_csv):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.

    with open(qc_with_paths_csv, "r") as f:
        candidate_info_list = []
        for row in list(csv.reader(f))[1:]:
            smri_path_str = row[0]
            target = row[1]
            if not target.lstrip("-").replace('.', '', 1).isdigit():
                continue
            motion_q_cscore_float = float(target)

            candidate_info_list.append(CandidateInfoTuple(
                smri_path_str,
                motion_q_cscore_float,
            ))

    candidate_info_list.sort(reverse=True, key=attrgetter('rating_int'))

    return candidate_info_list


def normalize_array(array):
    new_array = (array - array.min()) / (array.max() - array.min())

    return new_array


class Mri:
    def __init__(self, mri_path):
        image = tio.ScalarImage(mri_path)
        image_tensor = image.data
        image_tensor = image_tensor.squeeze()
        image_tensor = normalize_array(image_tensor)
        mri_a = np.array(image_tensor, dtype=np.float32)

        assert not np.any(np.isnan(mri_a))

        self.hu_a = mri_a

    def get_raw_candidate(self):
        mri_chunk = self.hu_a

        return mri_chunk


@functools.lru_cache(1, typed=True)
def get_mri(series_uid):
    return Mri(series_uid)


@raw_cache.memoize(typed=True)
def get_mri_raw_candidate(series_uid):
    mri = get_mri(series_uid)
    mri_chunk = mri.get_raw_candidate()
    return mri_chunk
