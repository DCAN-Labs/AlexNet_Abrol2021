import sys
import os
import argparse

import torch
import nibabel as nib
import numpy as np
sys.path.append('../..')
from reprex.models import AlexNet3D_Dropout_Regression
from util.logconf import logging
import json
import pdb

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def main():

    # book keeping and global variables
    parser = argparse.ArgumentParser(description='Script that controls BIDS conversion for individual studies.')
    parser.add_argument('--output_dir', help="The directory that the BIDS data will be outputted to.",required=True)
    parser.add_argument('--model_file', help="The path to the pre-trained model file",required=True)
    parser.add_argument('--input_file', help="The path to the image file that inference will be performed on.",required=True)
    args = parser.parse_args()

    # validate variables
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except:
            print('Could not create {out_dir}. Exiting...'.format(out_dir=args.output_dir))
            exit()
    assert os.path.exists(args.input_file), '{in_file} does not exist. Exiting...'.format(in_file=args.input_file)
    assert os.path.exists(args.input_file), '{model_file} does not exist. Exiting...'.format(model_file=args.model_file)

    # produce predictions
    pred = infer_bcp_motion_qc_score(args.model_file, args.input_file,args.output_dir)
    dictionary = {'prediction': pred}
    json_object = json.dumps(dictionary)
    with open(os.path.join(args.output_dir,'prediction.json'),'w') as outfile:
        outfile.write(json_object)
    log.info(f'prediction: {pred}')

def infer_bcp_motion_qc_score(model_weights_file, mri_path,output_dir):
    model = AlexNet3D_Dropout_Regression()
    model.load_state_dict(torch.load(model_weights_file,
                                     map_location='cpu'))
    model.eval()
    with torch.no_grad():
        prediction = get_prediction(model,mri_path,output_dir)
        return prediction

def get_prediction(model, mri_path, output_dir):
    
    filename=os.path.basename(mri_path).split('.nii.gz')[0]
    
    # TODO once Paul figures out issue with model training on infant MNI template resampled data the lines below will need to change
    new_mri_path=output_dir+'/'+filename+'_space-individual_den-BCP.nii.gz'
    
    # outside container
    os.system('flirt -in {mri_path} -ref /panfs/jay/groups/6/faird/shared/projects/motion-QC-generalization/code/AlexNet_Abrol2021/data/BCP/sub-380510_ses-20mo_run-001_T1w.nii.gz -out {new_mri_path} -applyxfm -init /panfs/roc/msisoft/fsl/6.0.1/etc/flirtsch/ident.mat'.format(mri_path=mri_path,new_mri_path=new_mri_path))

    # within container
    #os.system('flirt -in {mri_path} -ref /code/data/BCP/sub-380510_ses-20mo_run-001_T1w.nii.gz -out {new_mri_path} -applyxfm -init /opt/fsl-6.0.5.1/etc/flirtsch/ident.mat'.format(mri_path=mri_path,new_mri_path=new_mri_path))
    
    mri_nii_gz = nib.load(new_mri_path)
    mri_a = np.array(mri_nii_gz.get_fdata().copy(), dtype=np.float32)
    
    candidate_t = torch.from_numpy(mri_a).to(torch.float32)
    candidate_t = torch.unsqueeze(candidate_t.unsqueeze(0), dim=0)
    
    prediction_t_a = model(candidate_t)
    prediction = prediction_t_a[0].item()
    
    return prediction

if __name__ == "__main__":
    main()
    
    
