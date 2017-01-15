# set up all imports
import sys
sys.modules[__name__].__dict__.clear()

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import mne
from mne.io import Raw
from mne.beamformer import lcmv
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from mne import read_evokeds
from mne.minimum_norm import apply_inverse, read_inverse_operator


print(__doc__)
# I have set it up such that the MNE-sample-data is in the home directory..
data_path = '/home/parham/Desktop/MNE-sample-data'
# raw data file
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
subj = 'sample'
subjects_dir = data_path + '/subjects'
aseg_fname = subjects_dir + '/sample/mri/aseg.mgz'
inner_skull_surf_filename = '/home/parham/Desktop/MNE-sample-data/subjects/sample/bem/inner_skull.surf'
#save the solution to personal code directory.
dest_bem_dir='/home/parham/Desktop/python_code/sample-5120-5120-5120-bem.fif'
bem_sol_file='/home/parham/Desktop/python_code/sample-5120-5120-5120-bem-sol.fif'

fwd_sol_file='/home/parham/Desktop/python_code/sample_audvis-eeg-vol-fwd.fif'

inv_op_file='/home/parham/Desktop/python_code/sample_audvis-eeg-vol-inv.fif'
##################### MAke EEG inverse operator############


# Setup for reading the raw data
raw = Raw(raw_fname)
raw.info['bads'] = ['EEG 053']  # 2 bads channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=True,
eog=True,exclude='bads')
events = mne.read_events(event_fname)
reject = dict(eog=150e-6)
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin,
tmax=tmax, proj=True, picks=picks, baseline=(None, 0),
                      preload=True, reject=reject)
evoked = epochs.average()
#Make the BEM model
model =
mne.make_bem_model('/home/parham/Desktop/MNE-sample-data/subjects/sample')
mne.write_bem_surfaces(dest_bem_dir, model)
#setup a volume source space in the inner_skull. Compute BEM and store
result.
inner_skull_surface=mne.read_surface(inner_skull_surf_filename)
bem_sol = mne.make_bem_solution(model)
mne.write_bem_solution(bem_sol_file, bem_sol)
vol=mne.setup_volume_source_space(subj,surface='/home/parham/Desktop/MNE-sample-data/subjects/sample/bem/inner_skull.surf',
subjects_dir=subjects_dir)

# *************************MATLAB: Some simple
verification****************
# USE MATLAB to check that the volume source space has been setup
correctly.
vertidx = np.where(vol[0]['inuse'])[0]
pnts=vol[0]['rr']
closed_surface=inner_skull_surface[0]
pnts=pnts[vertidx]
import scipy.io as sio
sio.savemat('/home/parham/Desktop/python_code/src.mat',
{'pnts':pnts,'surf':closed_surface})

##### Returning to building the forward model and inverse operator
below:
#only compute the forward problem for EEG.
forward_eeg=mne.make_forward_solution(raw_fname, trans, vol,
bem_sol_file, fname=fwd_sol_file, meg=False, eeg=True, mindist=0.0,
ignore_ref=False,overwrite=True, n_jobs=1, verbose=None)
reject = dict(eeg=80e-6, eog=150e-6)
# Compute the covariance from the raw data
noise_cov = mne.compute_raw_covariance(raw, picks=picks, reject=reject)

info = evoked.info
inverse_operator_eeg = mne.minimum_norm.make_inverse_operator(info,
forward_eeg, noise_cov,loose=0.2, depth=0.8)
mne.minimum_norm.write_inverse_operator(inv_op_file,
inverse_operator_eeg)
# Read forward model from hard drive so that you check it has been
written.
forward = mne.read_forward_solution(fwd_sol_file)
# compute data covariance matrix from epochs.
data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15,
                                   method='shrunk')

#estimate VOLUME source
stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01,
pick_ori=None)

# Save result in stc files
stc.save('lcmv-vol')

stc.crop(0.0, 0.2)
src=forward['src']
# Save result in a 4D nifti file
img = mne.save_stc_as_volume('lcmv_inverse.nii.gz', stc,src,
mri_resolution=False)

t1_fname = data_path + '/subjects/sample/mri/T1.mgz'

# Plotting with nilearn
######################################################
plot_stat_map(index_img(img, 61), t1_fname, threshold=0.8,
               title='LCMV (t=%.1f s.)' % stc.times[61])

# plot source time courses with the maximum peak amplitudes
plt.figure()
plt.plot(stc.times, stc.data[np.argsort(np.max(stc.data,
axis=1))[-40:]].T)
plt.xlabel('Time (ms)')
plt.ylabel('LCMV value')
plt.show()
