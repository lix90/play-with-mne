import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.channels import read_montage
from mne.preprocessing import compute_proj_eog

# --------------- prepare directory ---------------
fext = '.vhdr'
data_path = os.path.expanduser('~') + '/toolbox/mne-demo/mne-data/'
fname = filter(lambda x: x.endswith(fext), os.listdir(data_path))

# --------------- preprocessing ---------------
if fext is '.set':
    raw = mne.io.read_raw_brainvision(os.path.join(data_path, fname[0]),
                                      eog=('VEO', 'HEOR'),
                                      preload=True)
elif fext is '.vhdr':
    raw = mne.io.read_raw_eeglab(os.path.join(data_path, fname[0]),
                                 eog=('VEOU', 'HEOL'),
                                 preload=True)


# raw = mne.add_reference_channels(raw, 'FCz', copy=False)
montage = read_montage('standard_1020')
raw.set_montage(montage)
ch_exclude = ['VEOD', 'HEOR', 'M2']

raw.pick_types(eeg=True, eog=True, stim=True, exclude=ch_exclude)
raw.pick_types(eeg=True, eog=True, stim=True)

raw_no_ref, _ = mne.io.set_eeg_reference(raw, [])
del raw

projs, events = compute_proj_eog(raw_no_ref, n_eeg=1, average=True)
raw_no_ref.info['projs'] = projs

raw_filtered = raw_no_ref.filter(0.1, 35, l_trans_bandwidth='auto',
                                 h_trans_bandwidth='auto')
raw_filtered.resample(125)
picks = mne.pick_types(raw_filtered.info, eeg=True, eog=False, stim=False)
events = mne.find_events(raw_filtered)
events_id = {'f/neg': 11, 's/neg': 22,
             'f/neu': 33, 'r/neg': 44,
             'sr/neg': 55}
epochs_no_proj = mne.Epochs(raw_filtered,
                            events=events,
                            event_id=events_id,
                            tmin=-0.2,
                            tmax=4.0,
                            proj=False,
                            baseline=(None, 0),
                            picks=picks,
                            reject=dict(eeg=300e-6),
                            preload=True,
                            detrend=1)
epochs_clean = mne.Epochs(raw_filtered,
                          events=events,
                          event_id=events_id,
                          tmin=-0.2,
                          tmax=4.0,
                          proj=True,
                          baseline=(None, 0),
                          picks=picks,
                          reject=dict(eeg=120e-6),
                          preload=True,
                          detrend=1)
epochs_clean.info['custom_ref_applied'] = False
epochs_clean.set_eeg_reference()

# plot epochs data
pz = epochs_clean.copy().pick_types(include=['Pz'])
fneg_pz = pz['f/neg'].average()
fneu_pz = pz['f/neu'].average()
diff_pz = fneg_pz - fneu_pz
lines = plt.plot(pz.times, fneg_pz.data.transpose(), 'b',
                 pz.times, fneu_pz.data.transpose(), 'k',
                 pz.times, diff_pz.data.transpose(), 'r')
plt.ylabel('Amptitude (uV)')
plt.xlabel('Times (ms)')
plt.setp(lines, linewidth=3, alpha=0.8)
plt.xlim((-0.1, 4.0))
locs, ticks = plt.yticks()
tmp = np.int8(np.round(locs * 1e6))
plt.yticks(locs, tuple([str(i) for i in tmp]))
plt.ylim((tmp.min() / 1e6, tmp.max() / 1e6))
lines_ = plt.plot([-0.1, 4.0], [0, 0], ':k',
                  [0, 0], [tmp.min()/1e6, tmp.max()/1e6], ':k')
plt.setp(lines_, linewidth=2, alpha=0.5)
plt.show()
