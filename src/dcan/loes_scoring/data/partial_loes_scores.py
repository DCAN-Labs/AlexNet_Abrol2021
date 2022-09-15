import csv
from datetime import datetime

from dcan.loes_scoring.data.AnteriorTemporalWhiteMatter import AnteriorTemporalWhiteMatter
from dcan.loes_scoring.data.LoesScore import LoesScore
from dcan.loes_scoring.data.ParietoOccipitalWhiteMatter import ParietoOccipitalWhiteMatter

file_path = \
    '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring/9_7 MRI sessions Igor Loes score updated.csv'
with open(file_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        print(', '.join(row))
        sub_id = row[0]
        sub_session = row[1]
        date_time_str = row[2]
        date_of_mri = datetime.strptime(date_time_str, '%m/%d/%Y')
        parieto_occipital_white_matter = \
            ParietoOccipitalWhiteMatter(
                periventricular=int(row[3]), central=int(row[4]), subcortical=int(row[5]), atrophy=int(row[6]))
        anterior_temporal_white_matter = \
            AnteriorTemporalWhiteMatter(
                periventricular=int(row[7]), central=int(row[8]), subcortical=int(row[9]), atrophy=int(row[10]))
        frontal_white_matter = None
        corpus_callosum = None
        visual_pathways = None
        auditory_pathway = None
        frontopontine_and_corticopsinal_fibers = None
        cerebellum = None
        white_matter_cerebellum_atrophy = None
        basal_ganglia = None
        anterior_thalamus = None
        global_atrophy = None
        brainstem_atrophy = None
        loes_score = None
        retricted_diffusion_present_on_mri = None
        gad = None
        loes_score_obj = \
            LoesScore(sub_id, sub_session, date_of_mri, parieto_occipital_white_matter, anterior_temporal_white_matter,
               frontal_white_matter, corpus_callosum, visual_pathways, auditory_pathway,
               frontopontine_and_corticopsinal_fibers, cerebellum, white_matter_cerebellum_atrophy, basal_ganglia,
               anterior_thalamus, global_atrophy, brainstem_atrophy, loes_score, retricted_diffusion_present_on_mri,
               gad)