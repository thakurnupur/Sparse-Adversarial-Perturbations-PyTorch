import argparse
import numpy as np
import pdb 
import torch, torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import *
from models import *

#Select the device to run the code 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ('Device: ' + str(device))

def calc_gradients(
		test_dataloader,
		model,
		max_iter,
		learning_rate,
		targets=None,
		weight_loss2=1,
		batch_size=1,
		seq_len=40):
	
	#Define the modifier and the optimizer
	modif = torch.Tensor(1, seq_len, 3, 112, 112).fill_(0.01/255).to(device)
	modifier = torch.nn.Parameter(modif, requires_grad=True)
	optimizer = torch.optim.Adam([modifier], lr=learning_rate)
	min_loss = 1e-5
	prev_loss = 1e-5

	for batch_index, (input_image, input_label) in enumerate(test_dataloader):

		input_image, input_label = input_image.to(device), input_label.to(device)

		#Clean video prediction 
		print(f'Batch Number: {batch_index}/{len(test_dataloader)}')
		print('------------------prediction for clean video-------------------')
		input_image = Variable(input_image, requires_grad=True)
		probs, pre_label = model(input_image) 
		print (f'Prediction: {pre_label.cpu().numpy()}, Original_label: {input_label.cpu().numpy()}')

		print('------------------prediction for adversarial video-------------------')

		min_in = input_image.min().detach()
		max_in = input_image.max().detach()

		all_loss = []
		for iiter in range(max_iter):
			
			input_image = Variable(input_image, requires_grad=False)

			model.lstm.reset_hidden_state()
			
			#Frames to be perturbed
			indicator = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0]
			
			#Perturbating the frames
			true_image = torch.clamp ((modifier[0,0,:,:,:]+input_image[0,0,:,:,:]), min_in, max_in)
			true_image = torch.unsqueeze(true_image, 0)
			
			for ll in range(seq_len-1):
				if indicator[ll+1] == 1:
					mask_temp = torch.clamp((modifier[0,ll+1,:,:,:]+input_image[0,ll+1,:,:,:]), min_in, max_in)
				else:
					mask_temp = input_image[0,ll+1,:,:,:]
				mask_temp = torch.unsqueeze(mask_temp,0)
				true_image = torch.cat((true_image, mask_temp),0)
			true_image = torch.unsqueeze(true_image, 0)
			
			for kk in range(batch_size-1):
				
				true_image_temp = torch.clamp((modifier[0,0,:,:,:]+input_image[kk+1,0,:,:,:]), min_in, max_in)
				true_image_temp = torch.unsqueeze(true_image_temp, 0)

				for ll in range(seq_len-1):
					if indicator[ll+1] == 1:
						mask_temp = torch.clamp((modifier[0,ll+1,:,:,:]+input_image[kk+1,ll+1,:,:,:]), min_in, max_in)
					else:
						mask_temp = input_image[kk+1,ll+1,:,:,:]
					mask_temp = torch.unsqueeze(mask_temp,0)
					true_image_temp = torch.cat((true_image_temp, mask_temp),0)
				true_image_temp = torch.unsqueeze(true_image_temp, 0)

				true_image = torch.cat((true_image, true_image_temp),0)

			#Prediction on the adversarial video
			probs, pre_label = model(true_image)

			#extracting te probability of true label 
			zero_array = torch.zeros(101).to(device)
			zero_array[input_label.cpu()] = 1
			true_label_onehot = probs*zero_array
			true_label_prob = torch.sum(true_label_onehot, 1)
			
			#Loss
			if targets is None:
			  loss1 = -torch.log(1 - true_label_prob + 1e-6)
			else:
			  loss1 = -torch.log(true_label_prob + 1e-6)
			loss1 = torch.mean(loss1)
			
			loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow((true_image-input_image), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
			norm_frame = torch.mean(torch.abs(modifier), dim=3).mean(dim=3).mean(dim=2) 

			loss = loss1 + weight_loss2 * loss2

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if iiter % 100 == 0: 
				print (f'Probability for ground truth label : {true_label_prob.detach().cpu().numpy()}')
				if prev_loss < loss : 
					print(f'Iteration: [{iiter}/{max_iter}], Loss: {loss}(\u25b2), Loss1: {loss1}, Loss2: {loss2}')
				elif prev_loss > loss: 
					print(f'Iteration: [{iiter}/{max_iter}], Loss: {loss}(\u25bc), Loss1: {loss1}, Loss2: {loss2}')
				else: 
					print(f'Iteration: [{iiter}/{max_iter}], Loss: {loss}, Loss1: {loss1}, Loss2: {loss2}')
			prev_loss = loss

			break_condition = False
			if loss < min_loss:
				if torch.abs(loss-min_loss) < 0.0001:
				   break_condition = True
				   print ('Aborting early!')
				min_loss = loss
			
			if iiter + 1 == max_iter or break_condition:
				print ('Norm frame for each frame: ')
				for pp in range(seq_len):
					# print the map value for each frame
					print(str(pp) + ' ' + str((norm_frame[0][pp]).detach().cpu().numpy()))

			print (f'Prediction for adversarial video: {pre_label.cpu().numpy()}, Original_label: {input_label.cpu().numpy()}')

			# Empty cache
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description='Sparse Adversarial Perturbations')
	parser.add_argument('-i', '--input_dir', type=str, default="", required=True,
						help='Directory of dataset.')
	parser.add_argument('--num_iter', type=int, default=100,
						help='Number of iterations to generate attack.')
	parser.add_argument('--learning_rate', type=float, default=0.001,  
						help='Learning rate of each iteration.')
	parser.add_argument('--target', type=str, default=None,
						help='Target list of dataset.')
	parser.add_argument('--weight_loss2', type=float, default=1,
						help='Weight of distance penalty.')
	parser.add_argument("--split_path", type=str, default="data/ucfTrainTestlist", help="Path to train/test split")
	parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
	parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
	parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
	parser.add_argument("--checkpoint_model", type=str, default="", help="Path to the checkpoint of the target model")

	parser.set_defaults(use_crop=True)
	args = parser.parse_args()
	print (args)

	seq_len = 40 #number of frames in a video
	batch_size = 1
	targets = None
	if args.target is not None:
		targets = {}
		with open(args.target, 'r') as f:
			for line in f:
				key, value = line.strip().split()
				targets[key] = int(value)

	image_shape = (args.channels, args.img_dim, args.img_dim)

	# Define test set   
	test_dataset = Dataset(dataset_path=args.input_dir,
		split_path=args.split_path,
		split_number=args.split_number,
		input_shape=image_shape,
		sequence_length=seq_len,
		training=False,
	)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	#Define the target classifier and load its checkpoint
	model = OrigConvLSTM(
		num_classes=101,
		latent_dim=512,
		lstm_layers=1,
		hidden_dim=1024,
		bidirectional=False,
	)

	if args.checkpoint_model:
		model.load_state_dict(torch.load(args.checkpoint_model))
	
	model = model.to(device)
	model.train()
	for m in model.modules():
		if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
			m.eval()
	print ('Model Loaded Successfully!')

	#Call the function to generate the adversarial videos
	calc_gradients(
	test_dataloader,
	model,
	args.num_iter,
	args.learning_rate,
	targets,
	args.weight_loss2,
	batch_size,
	seq_len)
				
if __name__ == '__main__':
	main()