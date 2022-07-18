import unittest

from dcan.loes_scoring.dsets import get_candidate_info_list, LoesScoreMRIs, get_loes_score_mris, \
    get_mri_raw_candidate, LoesScoreDataset


class TestDSets(unittest.TestCase):
    def test_get_candidate_info_list(self):
        candidate_info_list = get_candidate_info_list()
        self.assertIsNotNone(candidate_info_list)

    def test_LoesScoreMRIs_init(self):
        loes_score_mris = LoesScoreMRIs('5772LAVA_20180828')
        self.assertIsNotNone(loes_score_mris)
        raw_candidate = loes_score_mris.get_raw_candidate()
        self.assertIsNotNone(raw_candidate[0])
        self.assertIsNotNone(raw_candidate[0])

    def test_get_loes_score_mris(self):
        loes_score_mris = get_loes_score_mris('5772LAVA_20180828')
        self.assertIsNotNone(loes_score_mris)

    def test_get_mri_raw_candidate(self):
        mri_raw_candidate = get_mri_raw_candidate('5772LAVA_20180828')
        self.assertIsNotNone(mri_raw_candidate[0])
        self.assertIsNotNone(mri_raw_candidate[0])

    def test_loes_score_dataset_init(self):
        loes_score_dataset = LoesScoreDataset(val_stride=5)
        self.assertIsNotNone(loes_score_dataset)
        length = loes_score_dataset.__len__()
        self.assertEqual(4, length)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool,
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )

        return candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)


if __name__ == '__main__':
    unittest.main()
