import csv
import os

bcp_folder = 'data/BCP'
with open(os.path.join(bcp_folder, 'qc_with_paths.csv')) as csv_file_in:
    csv_reader = csv.reader(csv_file_in)
    next(csv_reader)
    with open(os.path.join(bcp_folder, 'qc_with_paths_standardized.csv'), 'w') as csv_file_out:
        csv_writer = csv.writer(csv_file_out, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['image_path', 'rating'])
        for row in csv_reader:
                csv_writer.writerow([os.path.join('/home/elisonj/shared/BCP/raw/BIDS_output', row[6]), row[7]])
