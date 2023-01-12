import csv
import math
import sys

import torch

from dcan.inference.infer import get_prediction
from reprex.models import AlexNet3DDropoutRegression
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def infer_multiple_files(model_weights_fl, csv_file):
    model = AlexNet3DDropoutRegression(3456)
    model.load_state_dict(torch.load(model_weights_fl,
                                     map_location='cpu'))
    model.eval()
    with torch.no_grad():
        with open(csv_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            count = 0
            total = 0.0
            distributions = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            for row in csv_reader:
                image_path = row[0]
                rating_str = row[1]
                if rating_str == 'NA':
                    continue
                rating = int(round(float(rating_str)))
                try:
                    prediction = get_prediction(model, image_path)
                except FileNotFoundError:
                    print("File not found.")
                    continue
                se = (rating - prediction) ** 2
                total += se
                count += 1
                distributions[rating].append(prediction)
            mse = total / count
            rmse = math.sqrt(mse)
            log.info(f'rmse: {rmse}')
    log.info(distributions)


if __name__ == "__main__":
    infer_multiple_files(sys.argv[1], sys.argv[2])
