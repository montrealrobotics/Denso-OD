'''
Preprocessing the image before passing as input
'''

from torchvision import transforms as T
import torch

def preprocess_image(cfg,img):

	"""
	Input: 
	cfg: configuration params
	img: Image loaded using matplotlib, numpy array of size H x W x C, in RGB format.
	(Note: if you use opencv to load image, convert it to RGB, as OpenCV works with BGR format)

	Output:
	torch tensor of size 1xCxHxW
	"""

	'''
	### ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor 
	of shape (C x H x W) in the range [0.0, 1.0]

	'''

	transform = T.Compose([ 
		T.ToTensor(),			
		T.Normalize(
		mean=list(cfg.INPUT.MEAN),
		std=list(cfg.INPUT.STD)
		)])

	return torch.unsqueeze(transform(img), 0) ## Appends additional dimension to convert into batch format.


