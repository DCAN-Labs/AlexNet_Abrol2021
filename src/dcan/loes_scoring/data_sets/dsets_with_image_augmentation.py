import copy
import csv
from datetime import datetime
import functools
import glob

from dcan.loes_scoring.data_sets.dsets import LoesScoreDataset, get_uid, CandidateInfoTuple, get_subject_session_info


@functools.lru_cache(1)
def get_candidate_augmentation_info_list(require_on_disk_bool=True):
    # We construct a set with all ald_code_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    loes_score_augmented_images_path = '/home/feczk001/shared/data/loes_scoring/Loes_score_augmented/sub-*/ses-*/'
    nifti_ext = '.nii.gz'
    mprage_mri_list = glob.glob(f'{loes_score_augmented_images_path}mprage_[0-9]{nifti_ext}')
    mprage_present_on_disk_set = {get_uid(p) for p in mprage_mri_list}
    present_on_disk_set = mprage_present_on_disk_set

    augmented_candidate_info_list = []
    scores_csv = '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring/augmented_files.csv'
    with open(scores_csv, "r") as f:
        for row in list(csv.reader(f))[1:]:
            session_str, subject_session_uid, subject_str, loes_score_str = get_subject_session_info(row)
            if subject_session_uid not in present_on_disk_set and require_on_disk_bool:
                continue
            if loes_score_str == '':
                continue
            loes_score_float = float(loes_score_str)
            session_date = datetime.strptime(session_str, '%Y%m%d')
            is_validated = False
            image_augmentation_index = int(row[4])
            augmented_candidate_info_list.append(CandidateInfoTuple(
                loes_score_float,
                subject_session_uid,
                subject_str,
                session_str,
                session_date,
                is_validated, image_augmentation_index
            ))

    augmented_candidate_info_list.sort(reverse=True)

    return augmented_candidate_info_list


class LoesScoreDatasetWithImageAugmentation(LoesScoreDataset):
    def __init__(self, val_stride=0, is_val_set_bool=None, subject=None, sortby_str='random'):
        super().__init__(val_stride, is_val_set_bool, subject, sortby_str)
        if not is_val_set_bool:
            # Cannot put augmented images in the training set if the original images are in the test set.  In other
            # words, can put an augmented image in the training set only if the original image is in the training set.
            candidate_augmentation_info_list = copy.copy(get_candidate_augmentation_info_list())
            for augmented_candidate in candidate_augmentation_info_list:
                in_original_training_set = False
                for candidate in self.candidateInfo_list:
                    if augmented_candidate.subject_str == candidate.subject_str and \
                            augmented_candidate.session_str == candidate.session_str:
                        in_original_training_set = True
                        break
                    if in_original_training_set:
                        break
                if in_original_training_set:
                    self.candidateInfo_list.append(augmented_candidate)
