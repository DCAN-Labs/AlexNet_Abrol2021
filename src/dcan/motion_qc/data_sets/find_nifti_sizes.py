import glob
import torchio as tio
import csv
with open('data/eLabe/qc_img_paths.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    t1_sizes = dict()
    t2_sizes = dict()
    next(csv_reader)
    for row in csv_reader:
        image_path = row[0]

        image = tio.ScalarImage(image_path)
        shape = image.shape
        if shape != (1, 208, 300, 320):
            print(f'row: {row}')
        if image_path.endswith('T1w.nii.gz'):
            if shape not in t1_sizes:
                t1_sizes[shape] = 0
            t1_sizes[shape] += 1
        elif image_path.endswith('T2w.nii.gz'):
            if shape not in t2_sizes:
                t2_sizes[shape] = 0
            t2_sizes[shape] += 1
print(f't1 sizes: {t1_sizes}')
print(f't2 sizes: {t2_sizes}')

# t1 sizes: {(1, 208, 300, 320): 722}
# t2 sizes: {(1, 208, 300, 320): 919, (1, 160, 300, 320): 1}
