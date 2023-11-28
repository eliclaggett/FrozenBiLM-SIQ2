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

3) Merge features

## Running
1) Test that the pretrained model can be evaluated on Social IQ2 \
`./run_eval.sh`

2) Fine-tune model
`./run_tune.sh`

3) Evaluate the fine-tuned model
`./run_eval_tuned.sh`