"""
-----------------------------------------------------------------------------------------
preproc_end_norm.py
-----------------------------------------------------------------------------------------
Goal of the script:
High-pass filter, z-score, average data and pick anat files
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject name
sys.argv[4]: group of shared data (e.g. 327)
-----------------------------------------------------------------------------------------
Output(s):
# Preprocessed and averaged timeseries files
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd ~/projects/gaze_exp/analysis_code/preproc/functional/
2. run python command
python preproc_filtering_normalization.py [main directory] [project name] [subject name] [group]
-----------------------------------------------------------------------------------------
Exemple:
python preproc_filtering_normalization.py /scratch/mszinte/data gaze_exp sub-001 327
-----------------------------------------------------------------------------------------
Written by Martin Szinte (mail@martinszinte.net)
-----------------------------------------------------------------------------------------
"""

# Stop warnings
import warnings
warnings.filterwarnings("ignore")

# General imports
import json
import sys
import os
import glob
import ipdb
import numpy as np
import nibabel as nb
import itertools as it
from nilearn import signal, masking
from nilearn.glm.first_level.design_matrix import _cosine_drift
from scipy.signal import savgol_filter
trans_cmd = 'rsync -avuz --progress'
deb = ipdb.set_trace

# Inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
group = sys.argv[4]

# load settings
with open('../../settings.json') as f:
    json_s = f.read()
    analysis_info = json.loads(json_s)
TR = analysis_info['TR']
high_pass_threshold = analysis_info['high_pass_threshold'] 
high_pass_type = analysis_info['high_pass_type'] 
session = 'ses-05'

# functional runs
pp_data_func_dir = "{}/{}/derivatives/pp_data/{}/func/desu/fmriprep_{}".format(main_dir, project_dir, subject, high_pass_type)

# Get fmriprep filenames
fmriprep_dir = "{}/{}/derivatives/fmriprep/fmriprep/{}/{}/func/".format(main_dir, project_dir, subject, session)

fmriprep_func_fns = glob.glob("{}/*_desc-preproc_bold.nii.gz".format(fmriprep_dir))
fmriprep_mask_fns = glob.glob("{}/*_desc-brain_mask.nii.gz".format(fmriprep_dir))



os.makedirs(pp_data_func_dir, exist_ok=True)

# High pass filtering and z-scoring
print("high-pass filtering...")
for func_fn, mask_fn in zip(fmriprep_func_fns,fmriprep_mask_fns):    
    masked_data = masking.apply_mask(func_fn, mask_fn)
    
    if high_pass_type == 'dct':
        n_vol = masked_data.shape[0]
        ft = np.linspace(0.5 * TR, (n_vol + 0.5) * TR, n_vol, endpoint=False)
        hp_set = _cosine_drift(high_pass_threshold, ft)
        masked_data = signal.clean(masked_data, detrend=False, standardize=True, confounds=hp_set)
        
    elif high_pass_type == 'savgol':
        window = int(np.round((1 / high_pass_threshold) / TR))
        masked_data -= savgol_filter(masked_data, window_length=window, polyorder=2, axis=0)
        masked_data = signal.clean(masked_data, detrend=False, standardize=True)
    
    high_pass_func = masking.unmask(masked_data, mask_fn)
    high_pass_func.to_filename("{}/{}_{}.nii.gz".format(pp_data_func_dir,func_fn.split('/')[-1][:-7],high_pass_type))
        
# ANATOMY
# Load anatomical data
print("getting anatomy...")
anat_session = 'ses-03'
output_files = ['dseg','desc-preproc_T1w','desc-aparcaseg_dseg','desc-aseg_dseg','desc-brain_mask']
orig_dir_anat = "{}/{}/derivatives/fmriprep/fmriprep/{}/{}/anat/".format(main_dir, project_dir, subject, anat_session)
dest_dir_anat = "{}/{}/derivatives/pp_data/{}/anat".format(main_dir, project_dir, subject)
os.makedirs(dest_dir_anat,exist_ok=True)

# Compute them
for output_file in output_files:
    orig_file = "{}/{}_{}_{}.nii.gz".format(orig_dir_anat, subject, anat_session, output_file)
    dest_file = "{}/{}_{}.nii.gz".format(dest_dir_anat, subject, output_file)
    os.system("{} {} {}".format(trans_cmd, orig_file, dest_file))
    
# Define permission cmd
os.system("chmod -Rf 771 {}/{}".format(main_dir, project_dir))
os.system("chgrp -Rf {} {}/{}".format(group, main_dir, project_dir))