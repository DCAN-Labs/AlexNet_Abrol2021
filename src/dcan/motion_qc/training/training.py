import torch
from torch.utils.data import DataLoader

from TrainingApp import TrainingApp, log
from dcan.motion_qc.data_sets.mri_motion_qc_score_dataset import MRIMotionQcScoreDataset
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

    def init_train_dl(self, data_csv):
        train_ds = MRIMotionQcScoreDataset(
            data_csv,
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
        val_ds = MRIMotionQcScoreDataset(csv_data_file,
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

    def get_output_distributions(self, csv_data_file):
        with torch.no_grad():
            val_dl = self.init_val_dl(csv_data_file)
            self.model.eval()
            batch_iter = enumerateWithEstimate(
                val_dl,
                "get_output_distributions",
                start_ndx=val_dl.num_workers,
            )
            distributions = dict()
            for batch_ndx, batch_tup in batch_iter:
                labels, n, predictions = self.get_labels_and_predictions(batch_tup)
                for i in range(n):
                    label_int = int(labels[i].item())
                    if label_int not in distributions:
                        distributions[label_int] = []
                    distributions[label_int].append(predictions[i])
            for distribution in distributions:
                distributions[distribution] = sorted(distributions[distribution])

        return distributions

    def get_standardized_rmse(self):
        with torch.no_grad():
            val_dl = self.init_val_dl(self.cli_args.csv_data_file)
            rmse, sigma = self.get_mean_and_sigma(val_dl)

            return rmse / sigma

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        csv_data_file = self.cli_args.csv_data_file
        train_dl = self.init_train_dl(csv_data_file)
        val_dl = self.init_val_dl(csv_data_file)

        self.run_epochs(train_dl, val_dl)

        try:
            standardized_rmse = self.get_standardized_rmse()
            log.info(f'standardized_rmse: {standardized_rmse}')
        except ZeroDivisionError as err:
            print('Could not compute stanardized RMSE because sigma is 0:', err)

        output_distributions = self.get_output_distributions(csv_data_file)
        log.info(f'output_distributions: {output_distributions}')

        torch.save(self.model.state_dict(), self.cli_args.model_save_location)


if __name__ == '__main__':
    InfantMRIMotionQCTrainingApp().main()
