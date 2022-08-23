import torchio as tio

file1 = open('/home/miran045/reine097/projects/AlexNet_Abrol2021/data/eLabe/possible_duplicates.txt', 'r')
Lines = file1.readlines()

duplicates = []


def get_image_data(image_file_path):
    image1 = tio.ScalarImage(image_file_path)
    data1 = image1.data
    data1 = data1.squeeze()
    data1 = data1.numpy()

    return data1


for line in Lines:
    ini_list = line.strip()
    res = ini_list.strip('][').split(', ')
    n = len(res)
    duplicates = []
    for i in range(n):
        data_1 = get_image_data(res[i])
        shape = data_1.shape
        for j in range(i + 1, n):
            data_2 = get_image_data(res[j])
            is_duplicate = False
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        if data_1[x][y][z] != data_2[x][y][z]:
                            is_duplicate = True
                            duplicates.append()
                            break
                    if is_duplicate:
                        break
                if is_duplicate:
                    break
            if is_duplicate:
                break



