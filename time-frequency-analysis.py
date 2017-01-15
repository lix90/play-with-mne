import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.preprocessing import (ICA, create_eog_epochs)
from mne.channels import (read_montage, read_dig_montage)
from mne.time_frequency import psd_multitaper

print(__doc__)

# --------------- prepare directory ---------------
fext = '.vhdr'
fpath = os.path.expanduser('~') + '/toolbox/mne-demo/mne-data/'
fdata = filter(lambda x: x.endswith(fext), os.listdir(fpath))

tmin, tmax = 0, 60 # 60s of data
fmin, fmax = 2, 100
n_fft = 2048

raw = mne.io.read_raw_brainvision(os.path.join(fpath, fdata[0]),
                                  eog=('VEO', 'HEOR'),
                                  preload=True)
raw.plot_psd(area_mode='range', tmax=10.0, show=False)

picks = mne.pick_types(raw.info, eeg=True, eog=False,
                       stim=False, exclude='bads')
picks = picks[:4]
plt.figure()
ax = plt.axes()
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
             n_jobs=1, proj=False, ax=ax, color=(0, 0, 1), picks=picks,
             show=False)

f, ax = plt.subplots()
psds, freqs = psd_multitaper(raw, low_bias=True, tmin=tmin, tmax=tmax,
                             fmin=fmin, fmax=fmax, proj=False, picks=picks,
                             n_jobs=1)
psds = 10*np.log10(psds)  # dB
psds_mean = psds.mean(0)
psds_std = psds.std(0)

ax.plot(freqs, psds_mean, color='k')
ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                color='k', alpha=0.5)

ax.set(title='Multitaper PSD', xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
plt.show()


raw = mne.add_reference_channels(raw, 'FCz', copy=False)
montage = read_montage('standard_1020')
raw.set_montage(montage)
raw.info['bads'] = ['TP9', 'TP10']  # bads channels or un-used
raw.info['custom_ref_applied'] = False
raw_no_ref, _ = mne.io.set_eeg_reference(raw, [])
picks = mne.pick_types(raw_no_ref.info,
                       eeg=True,
                       eog=False,
                       stim=False,
                       exclude='bads')

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
montage = read_montage('standard_1020')
montage_positions = montage.pos
montage_labels = montage.ch_names
montage_labels[2] = 'nasion'
digitization = read_dig_montage(hsp=montage_positions,
                                elp=montage_positions,
                                point_names=montage_labels,
                                unit='mm')
raw.set_montage(digitization)
raw.save()


# Calculate the centroid of all points in montage
x_pos = [p[1] for p in montage_positions]
y_pos = [p[0] for p in montage_positions]
z_pos = [p[2] for p in montage_positions]
montage_centroid = np.array([sum(x_pos) / len(x_pos),
                             sum(y_pos) / len(y_pos),
                             sum(z_pos) / len(z_pos)])

# filter & resample
raw_filtered = raw_car.filter(l_freq, h_freq,
                              l_trans_bandwidth='auto',
                              h_trans_bandwidth='auto').resample(125)

# --------------- ICA ---------------
picks = mne.pick_types(raw_filtered.info,
                       eeg=True,
                       eog=False,
                       stim=False,
                       exclude='bads')
rank_estimate = None
if rank_estimate is None:
    # estimate the rank only for the second VS task
    # use 300 seconds
    rank_estimate = raw_filtered.estimate_rank(tstart=240.,
                                               tstop=540.,
                                               picks=picks)
    print 'Estimated raw to be of rank', rank_estimate

    ica = ICA(n_components=rank_estimate,
              max_pca_components=None,
              max_iter=256,
              method='fastica',
              random_state=23)

    ica.fit(raw_filtered, picks=picks, decim=3)
    ica.plot_components()
    ica.plot_properties(raw_filtered, picks=1)

# --------------- EOG ---------------
eog_average = create_eog_epochs(raw_filtered,
                                picks=picks).average()

n_max_eog = 2
eog_epochs = create_eog_epochs(raw_filtered)
eog_inds, scores = ica.find_bads_eog(eog_epochs)

ica.plot_scores(scores, exclude=eog_inds)
ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course
ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})

ica.plot_overlay(eog_average, exclude=eog_inds, show=True)
ica.exclude.extend(eog_inds)
ica.save(fpath + '/demo-ica.fif')

# --------------- Apply ICA ---------------
l_freq_erp, h_freq_erp = 0.1, 40
l_trans_bandwidth = 'auto'
h_trans_bandwidth = 'auto'
raw_icaed = raw.filter(l_freq_erp,
                       h_freq_erp,
                       l_trans_bandwidth,
                       h_trans_bandwidth)
raw_icaed = ica.apply(raw_icaed,
                      exclude=eog_inds)

# --------------- Epoch ---------------
sfreq = 125
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
                    add_eeg_ref=False,
                    preload=True)
epochs.resample(sfreq)

# plot epochs data
picks_epochs = mne.pick_types(epochs.info, eeg=True)
fneg = epochs['f/neg'].average()
fneu = epochs['f/neu'].average()
epochs['f/neg'].plot_topo_image(vmin=-20, vmax=20, title='ERF images')
mne.combine_evoked([fneg, -fneu], weights='equal').plot_joint()
fneg.plot_joint()
