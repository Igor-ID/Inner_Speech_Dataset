from Python_Processing.Data_extractions import Extract_subject_from_BDF
from Python_Processing.Events_analysis import Check_Baseline_tags, Event_correction
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Channel names ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16',
# 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1',
# 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19',
# 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4',
# 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
# 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
# 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24',
# 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7',
# 'EXG8', 'Status']
eog = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
event_id = {'up': 31, 'down': 32, 'right': 33, 'left': 34}
# Number of session (1,2,3) 1-Silent speech, 2-Imagined speech, 3-Inner speech
N_B = 2
# Number of subject (1-10)
N_S = 2
root_dir = 'data'
# Get raw data
rawdata, N_sub = Extract_subject_from_BDF(root_dir, N_S, N_B)
print(rawdata.info)
# Re-reference. BioSemi is a “reference free” acquisition system, the Common-Mode (CM) voltage is recorded in all
# channels. For re-referencing, using channels EXG1 and EXG2 since they were placed in the left and right earlobe
Ref_channels = ['EXG1', 'EXG2']
rawdata.set_eeg_reference(ref_channels=Ref_channels)
print(rawdata.info)
# Apply band-stop filter (50Hz) and band-pass filter
Low_cut = 0.2
High_cut = 100
rawdata.notch_filter(freqs=50)
rawdata.filter(l_freq=Low_cut, h_freq=High_cut)
print(rawdata.info)
# Set channel types (EOG) to detect blinks, eye movements and face muscle activity
rawdata.set_channel_types({'EXG3': 'eog', 'EXG4': 'eog', 'EXG5': 'eog', 'EXG6': 'eog', 'EXG7': 'eog', 'EXG8': 'eog'})
# montage = mne.channels.make_standard_montage('biosemi128')
# rawdata.set_montage(montage)
print(rawdata.info)
events = mne.find_events(rawdata, initial_event=True, consecutive=True)
# Exclude spurious event
events = mne.pick_events(events, exclude=65536)

# In[ ] Epoching and decimating EEG
picks_eeg = mne.pick_types(rawdata.info, eeg=True, exclude=eog, stim=False)
epochsEEG = mne.Epochs(rawdata, events, event_id=event_id, tmin=-0.5, tmax=4,
                       picks=picks_eeg, preload=True, detrend=0, decim=4, baseline=None)

print(epochsEEG.info)
# The EEG and EXG channels for ICA
picks_vir = mne.pick_types(rawdata.info, eeg=True, include=eog, stim=False)
epochsEEG_full = mne.Epochs(rawdata, events, event_id=event_id, tmin=-0.5, tmax=4, picks=picks_vir, preload=True,
                            detrend=0, decim=4, baseline=None)
print(epochsEEG_full.info)


"""
Events_code = np.zeros([len(events[:, 2]), 2], dtype=int)
print(Events_code)
Events_code[:, 0] = events[:, 2]
print(Events_code)
Events_uniques = np.unique(Events_code[:, 0])
Event_count = np.zeros([len(Events_uniques), 2], dtype=int)
print(Event_count)
Event_count[:, 0] = Events_uniques
print(Event_count)
print(Events_uniques)
print(events[3, 2])
print(events[3, :])
print(len(events[:, -1]))
print(events[0:40, -1])
events_pr = Check_Baseline_tags(events)
print(events_pr[0:40, -1])
# print(events_pr[3, 2])
# print(events_pr[3, :])
print(len(events_pr[:, -1]))
events_cor = Event_correction(N_S=N_S, N_E=N_B, events=events)
events_cor_pr = Event_correction(N_S=N_S, N_E=N_B, events=events_pr)
print(events[270:300, -1])
print(events[380:410, -1])
print(events[885:915, -1])
print(events[335:395, -1])
print(events_cor[335:395, -1])
print(len(events_cor[:, -1]))
print(events_cor_pr[335:395, -1])
print(len(events_cor_pr[:, -1]))
mask = [61, 62, 63, 64]
res = np.argwhere(np.isin((events[:, -1]), mask)).ravel()
print(res)
# print(dir(rawdata))
# print(help(rawdata))
# print(dir(N_sub))
# print(N_sub)
# print(rawdata.__len__())
# print(rawdata.get_montage())
# print(rawdata.get_data())
# print(rawdata.ch_names)
plt.show()
"""
