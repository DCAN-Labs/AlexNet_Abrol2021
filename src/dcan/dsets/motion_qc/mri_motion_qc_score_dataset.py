import copy
import random

import torch
import torch.cuda
from torch.utils.data import Dataset

from dcan.dsets.motion_qc.dsets import getCandidateInfoList, getMriRawCandidate
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class MRIMotionQcScoreDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 is_val_set_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
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
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
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
        candidate_info_tup = self.candidateInfo_list[ndx]

        candidate_a = getMriRawCandidate(
            candidate_info_tup.series_uid,
        )
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        motion_qc_score_t = torch.tensor(candidate_info_tup.motionQCscore_int, dtype=torch.double)

        return candidate_t, motion_qc_score_t.float(), candidate_info_tup.series_uid
