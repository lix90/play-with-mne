import mne
import os
from mne.preprocessing import ICA
from mne.channels import read_montage

fext = '.set'  # or .vhdr
fpath = os.path.expanduser('~') + '/toolbox/mne-demo/mne-data/'
fdata = filter(lambda x: x.endswith(fext), os.listdir(fpath))

raw_or_epoch = 'raw'
sfreq = 250
lfreq_ica, hfreq_ica = 1, 40
decim_ica = 3
montage_type = 'standard_1020'
method_ica = 'fastica'
random_state = 24
reject_ica = dict(eeg=300e-6)

tmin_ep, tmax_ep = -0.2, 4.0
baseline = (None, None)
events_id = {'f/neg': 11, 's/neg': 22,
             'f/neu': 33, 'r/neg': 44,
             'sr/neg': 55}

if fext == '.set':
    # there are four EOG channels in the raw eeglab dataset,
    # When I made all four channels `eog` type
    # I found it confused the autodetection algorithm
    # It is strange, maybe just due to the low quality of ica
    ch_eog = ('VEOU', 'HEOL')
    ch_exclude = ['M2', 'VEOD', 'HEOR']
    read_raw = mne.io.read_raw_eeglab
elif fext == '.vhdr':
    # The montages are a little bit different
    # between the two kinds of datasets
    ch_eog = ('VEO', 'HEOR')
    ch_exclude = ['TP9', 'TP10']
    read_raw = mne.io.read_raw_brainvision

raw = read_raw(os.path.join(fpath, fdata[0]),
               eog=ch_eog,
               event_id=events_id,
               preload=True)

montage = read_montage(montage_type)
raw.set_montage(montage)
raw.pick_types(eeg=True, eog=True, stim=True, exclude=ch_exclude)
raw.info['custom_ref_applied'] = False

raw_filtered = raw.filter(lfreq_ica, hfreq_ica,
                          l_trans_bandwidth='auto',
                          h_trans_bandwidth='auto')
raw_filtered.resample(sfreq)

picks = mne.pick_types(raw.info, eeg=True, eog=False,
                       stim=False, exclude='bads')

events = mne.find_events(raw)
picks = mne.pick_types(raw_filtered.info,
                       eeg=True,
                       eog=False,
                       stim=False,
                       exclude='bads')
epochs = mne.Epochs(raw_filtered,
                    events=events,
                    event_id=events_id,
                    tmin=tmin_ep,
                    tmax=tmax_ep,
                    baseline=baseline,
                    picks=picks,
                    reject=reject_ica,
                    add_eeg_ref=False,
                    preload=True,
                    detrend=1)
picks_ep = mne.pick_types(epochs.info,
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
          method=method_ica,
          random_state=random_state)


if (raw_or_epoch == 'raw'):
    ica.fit(raw_filtered, picks=picks, decim=decim_ica, reject=reject_ica)
elif (raw_or_epoch == 'epoch'):
    ica.fit(epochs, picks=picks_ep, decim=decim_ica, reject=reject_ica)

# ica.plot_components()
