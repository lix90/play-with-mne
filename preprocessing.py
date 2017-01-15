import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.preprocessing import (ICA, create_eog_epochs)
from mne.channels import (read_montage, read_dig_montage)
from autoreject import LocalAutoRejectCV

# --------------- prepare directory ---------------
fext = '.set'
fpath = os.path.expanduser('~') + '/toolbox/mne-demo/mne-data/'
fdata = filter(lambda x: x.endswith(fext), os.listdir(fpath))


# --------------- preprocessing ---------------
events_id = {'f/neg': 11, 's/neg': 22,
             'f/neu': 33, 'r/neg': 44,
             'sr/neg': 55}
raw = mne.io.read_raw_eeglab(os.path.join(fpath, fdata[0]),
                             eog=('VEOU', 'HEOL'),
                             event_id=events_id,
                             preload=True)
# raw = mne.add_reference_channels(raw, 'FCz', copy=False)
montage = read_montage('standard_1020')
raw.set_montage(montage)
# raw.set_channels_type(mapping={'VEOD': 'eeg', 'HEOR': 'eeg'})
raw.info['bads'] = ['M2', 'VEOD', 'HEOR']
raw.info['custom_ref_applied'] = False
raw_no_ref, _ = mne.io.set_eeg_reference(raw, [])
# picks = mne.pick_types(raw_no_ref.info,
#                        eeg=True,
#                        eog=False,
#                        stim=False,
#                        exclude='bads')

# plot events
# mne.viz.plot_events(events, raw.info['sfreq'],
# raw.first_samp,
# event_id=events_id)
# plot raw data
# raw.plot(block=True)
# plot sensors
# raw.plot_sensors(kind='select',
#                  ch_type='all',
#                  title='standard_1020',
#                  show_names=True,
#                  block=True)
# plot psd
# raw.plot_psd(tmin=0,
#              tmax=100,
#              fmin=1,
#              fmax=80,
#              dB=True)

# Make digitization positions
# montage = read_montage('standard_1020')
# montage_positions = montage.pos
# montage_labels = montage.ch_names
# montage_labels[2] = 'nasion'
# digitization = read_dig_montage(hsp=montage_positions,
#                                 elp=montage_positions,
#                                 point_names=montage_labels,
#                                 unit='mm')
# raw.set_montage(digitization)
# raw.save()


# Calculate the centroid of all points in montage
# x_pos = [p[1] for p in montage_positions]
# y_pos = [p[0] for p in montage_positions]
# z_pos = [p[2] for p in montage_positions]
# montage_centroid = np.array([sum(x_pos) / len(x_pos),
#                              sum(y_pos) / len(y_pos),
#                              sum(z_pos) / len(z_pos)])

# filter & resample

# --------------- ICA ---------------
l_freq, h_freq, sfreq = 1, 40, 100
raw_filtered = raw_no_ref.filter(l_freq, h_freq,
                                 l_trans_bandwidth='auto',
                                 h_trans_bandwidth='auto').resample(sfreq)
picks = mne.pick_types(raw_filtered.info,
                       eeg=True,
                       eog=False,
                       stim=False,
                       exclude='bads')
rank_estimate = raw_filtered.estimate_rank(tstart=240.,
                                           tstop=540.,
                                           picks=picks)
ica = ICA(n_components=rank_estimate,
          max_pca_components=None,
          max_iter=256,
          method='fastica',
          random_state=24)

ica.fit(raw_filtered, picks=picks, decim=3, reject=dict(eeg=200e-6))
ica.plot_components()
ica.plot_properties(raw_filtered, picks=[0])
# --------------- EOG ---------------
eog_average = create_eog_epochs(raw_filtered,
                                picks=picks).average()

n_max_eog = 4
eog_epochs = create_eog_epochs(raw_filtered)
eog_inds, scores = ica.find_bads_eog(eog_epochs)

ica.plot_scores(scores, exclude=eog_inds)
ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course
ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})

ica.plot_overlay(eog_average, exclude=eog_inds, show=True)
ica.exclude.extend(eog_inds)
ica.save(fpath + '/demo-old-ica.fif')

# --------------- Apply ICA ---------------
l_freq_erp, h_freq_erp = None, 40
raw_icaed = raw_no_ref.filter(l_freq_erp,
                              h_freq_erp,
                              l_trans_bandwidth='auto',
                              h_trans_bandwidth='auto')
ica.apply(raw_icaed, exclude=eog_inds)
raw_icaed.info['custom_ref_applied'] = False
raw_icaed.set_eeg_reference(ref_channels=None)
raw_icaed.save(fpath + '/old_icaed_raw.fif', overwrite=True)
# --------------- Epoch ---------------
tmin, tmax = -0.2, 4.0
sfreq = 250
events = mne.find_events(raw_icaed)
picks = mne.pick_types(raw_icaed.info,
                       eeg=True,
                       eog=False,
                       stim=False,
                       exclude='bads')
epochs = mne.Epochs(raw_icaed,
                    events=events,
                    event_id=events_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=(None, 0),
                    picks=picks,
                    reject=dict(eeg=80e-6),
                    add_eeg_ref=True,
                    preload=True,
                    detrend=1).resample(sfreq)

# autoreject
ar = LocalAutoRejectCV()
epochs_clean = ar.fit_transform(epochs)

# plot epochs data
pz = epochs_clean.copy().pick_types(include=['POz'])
fneg_pz = pz['f/neg'].average()
fneu_pz = pz['f/neu'].average()
plt.plot(pz.times, fneg_pz.data.transpose(), 'b', hold=True)
plt.plot(pz.times, fneu_pz.data.transpose(), 'k')
plt.xlim((-0.2, 2.0))
plt.show()
