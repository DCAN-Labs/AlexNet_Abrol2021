import os

# r=root, d=directories, f = files
path = '/home/feczk001/shared/data/loes_scoring/Loes_score/'
for r, d, f in os.walk(path):
    for file in f:
        if 'mprage' not in file and file != 'loes_scores.csv':
            os.remove(os.path.join(r, file))
