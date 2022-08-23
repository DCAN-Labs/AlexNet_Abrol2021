import csv

import torchio as tio


def hash_code(data):
    shape = data.shape
    hash = 7
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                voxel = data.item((i, j, k))
                hash = (31 * hash + voxel) % 1024
    return hash

with open('/home/miran045/reine097/projects/AlexNet_Abrol2021/data/eLabe/qc_img_paths.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)
    hash_to_files = dict()
    for row in csv_reader:
        file_name = row[0]
        image = tio.ScalarImage(file_name)
        data = image.data
        data = data.squeeze()
        data = data.numpy()
        hc = hash_code(data)
        if hc not in hash_to_files:
            hash_to_files[hc] = []
        hash_to_files[hc].append(file_name)
        print(f'{hc}: {file_name}')
        if len(hash_to_files[hc]) > 1:
            print('Collision')
    for key in hash_to_files:
        files = hash_to_files[key]
        if len(files) > 1:
            print(files)
