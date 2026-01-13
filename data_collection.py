import time
import json
import pprint
import numpy as np
from InfineoenManager import InfineonManager as RadarManager
from processing_utils import spectrogram
from pathlib import Path
import argparse



n_frames = 20

# Define a dictionary mapping gestures to numeric labels
gesture_dict = {
    'SwipeLeft': 0,
    'SwipeRight': 1,
    'SwipeDown': 2,
    'SwipeUp': 3,
    'Push': 4
}

## Record Configuration
chirp_config_path = 'configs/cfg_simo_chirp.json'
seq_config_path = 'configs/cfg_simo_seq.json'

# chirp_config_path = 'configs/cfg_chirp.json'
# seq_config_path = 'configs/cfg_seq.json'

with open(chirp_config_path) as file:
    cfg_chirp = json.load(file)

with open(seq_config_path) as file:
    cfg_seq = json.load(file)

radar = RadarManager()
radar.init_device_fmcw(cfg_seq, cfg_chirp)

## Calculate Radar Parameters
params = radar.get_params({**cfg_chirp, **cfg_seq})
# print('Radar Parameters:')
# pprint.pprint(params)


## Save data
parser = argparse.ArgumentParser()
parser.add_argument("--person", default="Michael")
parser.add_argument("--gesture", required=True)
parser.add_argument("--id", type=int, required=True)
args = parser.parse_args()

person_name = args.person
gesture = args.gesture
id = args.id


raw_file_name = f"./data/train/raw_data/{person_name}/{gesture}/{id}.npy" # Make sure the directory exists
md_file_name = f"./data/train/spectrogram/{person_name}/{gesture}/{id}.png" # Make sure the directory exists


## Record
start_time = time.time()
data = radar.fetch_n_frames(n_frames)
print(f"Collected data in {time.time() - start_time:.2f} seconds")
print('data shape:', data.shape)  # (n_frame, n_antenna, n_chirp, n_sample)
Path(raw_file_name).parent.mkdir(parents=True, exist_ok=True)
np.save(raw_file_name, data)

## Spectrogram
Path(md_file_name).parent.mkdir(parents=True, exist_ok=True)
spectrogram(data, duration=data.shape[0]*cfg_seq['frame_repetition_time_s'], prf=params['prf'], mti=True, is_save=True, savename=md_file_name)
# spectrogram(data, duration=data.shape[0]*cfg_seq['frame_repetition_time_s'], prf=params['prf'], mti=True)