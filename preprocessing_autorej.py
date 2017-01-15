import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.preprocessing import (ICA, create_eog_epochs)
from mne.channels import read_montage
from mne.epochs import make_fixed_length_events
from autoreject import LocalAutoRejectCV, Ransac
import meeg_preprocessing as meeg

# --------------- prepare directory ---------------
fext = '.set'
data_path = os.path.expanduser('~') + '/toolbox/mne-demo/mne-data/'
fname = filter(lambda x: x.endswith(fext), os.listdir(data_path))

# --------------- preprocessing ---------------
raw = mne.io.read_raw_eeglab(os.path.join(data_path, fname[1]),
                             eog=('VEOU', 'HEOL'),
                             preload=True)
# raw = mne.add_reference_channels(raw, 'FCz', copy=False)
montage = read_montage('standard_1020')
raw.set_montage(montage)
ch_exclude = ['VEOD', 'HEOR', 'M2']
raw.pick_types(eeg=True, eog=True, stim=True, exclude=ch_exclude)
raw_no_ref, _ = mne.io.set_eeg_reference(raw, [])
del raw

raw_filtered = raw_no_ref.filter(1, 35, l_trans_bandwidth='auto',
                                 h_trans_bandwidth='auto')
raw_filtered.resample(125)
picks = mne.pick_types(raw_filtered.info, eeg=True, eog=False, stim=False)
events = mne.find_events(raw_filtered)
epochs = mne.Epochs(raw_filtered,
                    events=events,
                    tmin=-0.2,
                    tmax=4.0,
                    baseline=(None, None),
                    picks=picks,
                    reject=dict(eeg=300e-6),
                    preload=True,
                    detrend=1)

ica = ICA(n_components=0.99,
          method='fastica',
          random_state=23)
ica.fit(raw_filtered, picks=picks, decim=3, reject=None)
ica.plot_components()
ica.plot_properties(epochs,  picks=[0, 1, 2, 3, 4, 5])


subject = 'demo'


# autoreject EOG
eog_average = create_eog_epochs(raw_filtered,
                                picks=picks_ica).average()
n_max_eog = 4
eog_epochs = create_eog_epochs(raw_filtered)
eog_inds, scores = ica.find_bads_eog(eog_epochs)

ica.plot_scores(scores, exclude=eog_inds)
ica.plot_sources(eog_average, exclude=eog_inds)
ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})
ica.plot_overlay(eog_average, exclude=eog_inds, show=True)
ica.exclude.extend(eog_inds)
ica.save(fpath + '/demo_old_epoch_ica.fif')

# apply ICA
l_freq_erp, h_freq_erp = None, 40
raw_icaed = raw_no_ref.filter(l_freq_erp, h_freq_erp,
                              l_trans_bandwidth='auto',
                              h_trans_bandwidth='auto')
ica.apply(raw_icaed, exclude=eog_inds)
raw_icaed.info['custom_ref_applied'] = False
raw_icaed.set_eeg_reference(ref_channels=None)
raw_icaed.save(fpath + '/old_ica_raw.fif', overwrite=True)

# --------------- Epoch ---------------
tmin, tmax = -0.2, 4.0
sfreq = 250
events_id = {'f/neg': 11, 's/neg': 22,
             'f/neu': 33, 'r/neg': 44,
             'sr/neg': 55}
events = mne.find_events(raw_filtered)
picks = mne.pick_types(raw_filtered.info, eeg=True,
                       eog=False, stim=False,
                       exclude='bads')
epochs = mne.Epochs(raw_filtered,
                    events=events,
                    event_id=events_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=(None, None),
                    picks=picks,
                    reject=None,
                    preload=True,
                    detrend=1)

# automatic
ar = LocalAutoRejectCV(verbose='tqdm')
epochs_clean = ar.fit_transform(epochs)

picks_ica = mne.pick_types(epochs_clean.info, eeg=True,
                           eog=False, stim=False,
                           exclude='bads')
ica = ICA(n_components=0.99,
          max_pca_components=None,
          max_iter=256, method='fastica',
          random_state=24)
ica.fit(epochs_clean, picks=picks_ica, decim=3, reject=dict(eeg=200e-6))
ica.plot_components()
ica.plot_properties(epochs_clean, picks=[0, 1, 2, 3, 4, 5])
ica.exclude.extend([0, 4])
epochs_clean_ica = ica.apply(epochs_clean, exclude=[0, 4])
epochs_clean_ica.plot(n_epochs=5, n_channels=10, block=True)
# plot epochs data
pz = epochs_clean.copy().pick_types(include=['POz'])
fneg_pz = pz['f/neg'].average()
fneu_pz = pz['f/neu'].average()
diff_pz = fneg_pz - fneu_pz
lines = plt.plot(pz.times, fneg_pz.data.transpose(), 'b',
                 pz.times, fneu_pz.data.transpose(), 'k',
                 pz.times, diff_pz.data.transpose(), 'r')
plt.ylabel('Amptitude (' + r'$\muV$' + ')')
plt.xlabel('Times (ms)')
plt.setp(lines, linewidth=3, alpha=0.8)
plt.xlim((-0.1, 1.0))
locs, ticks = plt.yticks()
tmp = np.int8(np.round(locs*1e6))
plt.yticks(locs, tuple([str(i) for i in tmp]))
plt.ylim((tmp.min()/1e6, tmp.max()/1e6))
lines_ = plt.plot([-0.1, 1.0], [0, 0], ':k',
                  [0, 0], [tmp.min()/1e6, tmp.max()/1e6], ':k')
plt.setp(lines_, linewidth=2, alpha=0.5)

plt.show()
