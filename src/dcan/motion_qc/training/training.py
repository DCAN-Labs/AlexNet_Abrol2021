import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TrainingApp import TrainingApp, log
from dcan.loes_scoring.model.luna_model import LunaModel
from dcan.models.AlexNet3D_Dropout_Regression_deeper import AlexNet3D_Dropout_Regression_deeper
from dcan.motion_qc.data_sets.mri_motion_qc_score_dataset import MRIMotionQcScoreDataset
from reprex.models import AlexNet3D_Dropout_Regression
from util.util import enumerateWithEstimate


class InfantMRIMotionQCTrainingApp(TrainingApp):
    def __init__(self):
        super().__init__()

        self.cli_args = self.get_args()

        self.val_writer = None
        self.totalTrainingSamples_count = 0
        self.global_step_tr = 0
        self.global_step_val = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def init_train_dl(self):
        train_ds = MRIMotionQcScoreDataset(
            val_stride=10,
            is_val_set_bool=False)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def init_val_dl(self, csv_data_file=None):
        val_ds = MRIMotionQcScoreDataset(
            val_stride=10,
            is_val_set_bool=True)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def get_output_distributions(self):
        with torch.no_grad():
            val_dl = self.init_val_dl()
            self.model.eval()
            batch_iter = enumerateWithEstimate(
                val_dl,
                "get_output_distributions",
                start_ndx=val_dl.num_workers,
            )
            distributions = {1: [], 2: [], 3: [], 4: [], 5: []}
            for batch_ndx, batch_tup in batch_iter:
                labels, n, predictions = self.get_labels_and_predictions(batch_tup)
                for i in range(n):
                    label_int = int(labels[i].item())
                    distributions[label_int].append(predictions[i])
            for distribution in distributions:
                distributions[distribution] = sorted(distributions[distribution])

        return distributions

    def get_standardized_rmse(self):
        with torch.no_grad():
            val_dl = self.init_val_dl()
            rmse, sigma = self.get_mean_and_sigma(val_dl)

            return rmse / sigma

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        self.run_epochs(train_dl, val_dl)

        try:
            standardized_rmse = self.get_standardized_rmse()
            log.info(f'standardized_rmse: {standardized_rmse}')
        except ZeroDivisionError as err:
            print('Could not compute stanardized RMSE because sigma is 0:', err)

        output_distributions = self.get_output_distributions()
        log.info(f'output_distributions: {output_distributions}')

        torch.save(self.model.state_dict(), self.cli_args.model_save_location)


if __name__ == '__main__':
    InfantMRIMotionQCTrainingApp().main()
