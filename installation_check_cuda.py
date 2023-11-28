import torch

if torch.cuda.is_available():
    print('CUDA is available!')
    print('Using:',torch.cuda.get_device_name())
else:
    print('CUDA is not available!')
    print('Please check that CUDA is installed and that PyTorch has been configured to use it')