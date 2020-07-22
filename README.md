# Sparse-Adversarial-Perturbations-PyTorch
This is the PyTorch implementation of the paper '[Sparse Adversarial Perturbations for Videos](https://arxiv.org/pdf/1803.02536.pdf)'. 

This code requires Python3 and requirements.txt contains all its other dependencies. Run the following command to install all these requirements: 
```python
pip install -r requirements.txt
```

To run the code, use the following command: 
```python
python l21_optimization.py --input_dir data/UCF-101-frames --split_path data/ucfTrainTestlist --checkpoint_model ConvLSTM_150.pth
```

Please refer to the following link for the code for the processing of UCF-101 dataset and its target network: 
https://github.com/eriklindernoren/Action-Recognition

You can find the checkpoint model for CNN+LSTM model on this link too. 
