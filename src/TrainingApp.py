import argparse


class TrainingApp:
    def __init__(self):
        self.device = None
        self.model = None

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--num-workers',
                                 help='Number of worker processes for background data loading',
                                 default=8,
                                 type=int,
                                 )
        self.parser.add_argument('--batch-size',
                                 help='Batch size to use for training',
                                 default=32,
                                 type=int,
                                 )
        self.parser.add_argument('--epochs',
                                 help='Number of epochs to train for',
                                 default=1,
                                 type=int,
                                 )

    @staticmethod
    def get_actual(outputs):
        actual = outputs[0].squeeze(1)

        return actual

    def init_val_dl(self, csv_data_file):
        pass
