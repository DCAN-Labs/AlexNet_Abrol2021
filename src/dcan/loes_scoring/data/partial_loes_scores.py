import csv
from datetime import datetime

from dcan.loes_scoring.data.AnteriorTemporalWhiteMatter import AnteriorTemporalWhiteMatter
from dcan.loes_scoring.data.AuditoryPathway import AuditoryPathway
from dcan.loes_scoring.data.CorpusCallosum import CorpusCallosum
from dcan.loes_scoring.data.FrontalWhiteMatter import FrontalWhiteMatter
from dcan.loes_scoring.data.Frontopontine_And_Corticopsinal_Fibers import Frontopontine_And_Corticopsinal_Fibers
from dcan.loes_scoring.data.LoesScore import LoesScore
from dcan.loes_scoring.data.ParietoOccipitalWhiteMatter import ParietoOccipitalWhiteMatter
from dcan.loes_scoring.data.VisualPathways import VisualPathways

file_path = \
    '/home/miran045/reine097/projects/AlexNet_Abrol2021/data/loes_scoring/9_7 MRI sessions Igor Loes score updated.csv'
loes_scores = []
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
        frontal_white_matter = \
            FrontalWhiteMatter(
                periventricular=int(row[11]), central=int(row[12]), subcortical=int(row[13]), atrophy=int(row[13]))
        corpus_callosum = \
            CorpusCallosum(splenium=int(row[14]), body=int(row[15]), genu=int(row[16]), atrophy=int(row[17]))
        visual_pathways = \
            VisualPathways(
                optic_radiation=int(row[18]), meyers_loop=int(row[19]), lateral_geniculate_body=int(row[20]),
                optic_tract=int(row[21]))
        auditory_pathway = \
            AuditoryPathway(
                medial_geniculate=int(row[22]), brachium_to_inferior_colliculus=int(row[23]),
                lateral_leminiscus=int(row[24]), trapezoid_body_pons=int(row[25]))
        frontopontine_and_corticopsinal_fibers = \
            Frontopontine_And_Corticopsinal_Fibers(internal_capsule=int(row[26]), brain_stem=int(row[27]))
        cerebellum = int(row[28])
        white_matter_cerebellum_atrophy = int(row[29])
        basal_ganglia = int(row[30])
        anterior_thalamus = int(row[31])
        global_atrophy = int(row[32])
        brainstem_atrophy = int(row[33])
        loes_score = int(row[34])
        retricted_diffusion_present_on_mri = True if row[35] == 'Yes' else False
        gad = int(row[35])
        loes_score_obj = \
            LoesScore(sub_id, sub_session, date_of_mri, parieto_occipital_white_matter, anterior_temporal_white_matter,
               frontal_white_matter, corpus_callosum, visual_pathways, auditory_pathway,
               frontopontine_and_corticopsinal_fibers, cerebellum, white_matter_cerebellum_atrophy, basal_ganglia,
               anterior_thalamus, global_atrophy, brainstem_atrophy, loes_score, retricted_diffusion_present_on_mri,
               gad)
        loes_scores.append(loes_score_obj)
print(loes_scores)
