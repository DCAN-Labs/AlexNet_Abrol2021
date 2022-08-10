import unittest

from dcan.loes_scoring.data_sets.dsets_with_image_augmentation import get_candidate_augmentation_info_list, \
    LoesScoreDatasetWithImageAugmentation


class TestDSets(unittest.TestCase):
    def test_get_candidate_info_list(self):
        augmented_candidate_info_list = get_candidate_augmentation_info_list()
        self.assertIsNotNone(augmented_candidate_info_list)

    def test_loes_score_augmented_dataset_init(self):
        loes_score_dataset = LoesScoreDatasetWithImageAugmentation(val_stride=10)
        self.assertIsNotNone(loes_score_dataset)
        length = loes_score_dataset.__len__()
        self.assertEqual(450, length)
        for i in range(3):
            item = loes_score_dataset.__getitem__(i)
            self.assertIsNotNone(item)
        item = loes_score_dataset.__getitem__(length - 1)
        self.assertIsNotNone(item)


if __name__ == '__main__':
    unittest.main()
