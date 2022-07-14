from unittest import TestCase

from dcan.dsets.loes_scoring.dsets import get_candidate_info_list


class TestDSets(TestCase):
    def test_get_candidate_info_list(self):
        candidate_info_list = get_candidate_info_list(require_on_disk_bool=False)
        self.assertIsNotNone(candidate_info_list)
