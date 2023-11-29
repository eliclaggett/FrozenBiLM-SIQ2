# Frozen BiLM for Social IQ

## Installation

1) Install the required Python packages (PyTorch may need to be separately installed with CUDA enabled) \
`pip install -r requirements.txt`

2) Check that PyTorch is able to use CUDA \
`python3 installation_check_cuda.py`

3) Download the pretrained language model \
`python 3 installation_language_model.py`

4) Download a FrozenBiLM model checkpoint
Get the model checkpoint here: \
https://drive.google.com/file/d/15vsfNJf9UsWbmimibfPhLoHRrbZhF7W6/view \
Then, save it to "model/ckpt/frozenbilm_tvqa.pth"

### Extra Installation Steps for Social IQ2
1) Format the dataset into a train, val, and test CSV files

2) Extract features

`python3 extract/siq2_prep.py` set video path and desired output path for each video and save that in one big lookup table

`./installation_extract.sh` visual feats per second for each video, seperate files

3) Merge features


`./installation_merge.sh` all visual feats are 60x768 keyed by video id in one big clip.pth

`python3 ./installation_merge_face.py` facial feats per second for each video, one big face.pth

<!-- `python3 ./installation_merge_pose.py` pose feats per second for each video, one big pose.pth -->

<!-- MC_Dataset self.features is the dictionary of 60 frames for video, but we set max_feats to 10 so we sample 10 frames

in batch_dict, video is 1x10x768 -->

## Running
1) Test that the pretrained model can be evaluated on Social IQ2 \
`./run_eval.sh`

2) Fine-tune model
`./run_tune.sh`

3) Evaluate the fine-tuned model
`./run_eval_tuned.sh`