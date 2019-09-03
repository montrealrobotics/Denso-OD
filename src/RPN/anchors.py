'''
Base class to generate anchors. 
This code will mostly be used for training!(You never know though!)
'''

import numpy as np

## TODO: make proper hierarchy for the functions to be called in a particular order!!

class anchor_generator(object):
	"""docstring for anchor_generator"""
	def __init__(self):
		super(anchor_generator, self).__init__()
		## Do nothing!
	
	## TODO: Automate the process of getting these inputs. Make the inputs configuration parameters.

	def get_anchors(self, image, feature_map, cfg):
		
		'''

		Input: 
		image: A single image(torch tensor) of size BZxCxHxW, BS = batchsize, C = channels, H = height, W = width
		feature_map: Torch tensor of size BSxFcxFhxFw, BS = batchsize, Fc = feature map channels, Fh = feature map height, Fw = feature map width
		aspect_ratios: An array of various anchor ratios
		anchor_scales: An array of various anchor scales

		Output: 
		Nx4 numpy array of anchor locations. N = number of anchors. 
		
		'''
		self.image = image
		self.feature_map = feature_map
		self.subsample = [self.image.size()[2]//self.feature_map.size()[2],
							self.image.size()[3]//self.feature_map.size()[3]] ## Subsampling ratio. 
		self.aspect_ratios = list(cfg.ANCHORS.ASPECT_RATIOS)
		self.anchor_scales = list(cfg.ANCHORS.ANCHOR_SCALES)

		## Let's get anchor centres(wrt original image)
		return self.get_all_anchors(im_height = self.image.size()[2], 
									im_width = self.image.size()[3], 
									sub_sample = self.subsample,
									aspect_ratios = self.aspect_ratios,
									anchor_scales = self.anchor_scales)




	def get_valid_anchors(self):
		# self.anchor_data = self.get_all_anchors() ## TODO: Get all the necessary arguments
		valid_anchor_indices = np.where((self.anchor_data[:,0] >= 0) &
							 (self.anchor_data[:,1] >= 0) &
							 (self.anchor_data[:,2] <= im_height) &
							 (self.anchor_data[:,3] <= im_width))[0]

		# print(valid_anchor_indices.shape)


		## Constructing an array holding valid anchor boxes
		self.valid_anchor_boxes = self.anchor_data[valid_anchor_indices]

	def generate_anchors(self, anchor_center, aspect_ratios, anchor_scales, sub_sample):
	
		'''
		Input: 
		
		anchor_center: [y, x] ## Notation similar to how BB are defined. Co-ordinates in original image.
		aspect_ratios: [ar1, ar2, ar3]
		anchor_scales: [as1, as2, as3]
		
		Output: A numpy array of 9x4, generates 9 anchors with center as given anchor center

		Testing the function:

		anchor_center = [100, 100]
		aspect_ratios = [0.5, 1, 2]
		anchor_scales = [8, 16, 32]
		sub_sample = 16
		anchor_base = generate_anchors(anchor_center, aspect_ratios, anchor_scales, sub_sample)

		'''
		
		## We will have 9 anchors, each with 4 co-ordinates
		anchor_base = np.zeros((len(anchor_scales) * len(aspect_ratios), 4), dtype=np.float32)
		
		## Determining co-ordinates for all the anchors
		for i in range(len(anchor_scales)):
			for j in range(len(aspect_ratios)):
		   
				## height and width should be such that h/w = aspect_ratios[j]
				anchor_height = sub_sample[0]*anchor_scales[i]*np.sqrt(aspect_ratios[j])
				anchor_width = sub_sample[1]*anchor_scales[i]*np.sqrt(1.0/aspect_ratios[j])
				
				anchor_index = i*len(anchor_scales) + j 
				
				## y co-ordinate of top left corner
				anchor_base[anchor_index, 0] = anchor_center[0] - anchor_height/2.0
				## x co-ordinate of top left corner
				anchor_base[anchor_index, 1] = anchor_center[1] - anchor_width/2.0
				
				## y co-ordinate of bottom right corner
				anchor_base[anchor_index, 2] = anchor_center[0] + anchor_height/2.0
				## x co-ordinate of bottom right corner
				anchor_base[anchor_index, 3] = anchor_center[1] + anchor_width/2.0
				
				
		return anchor_base

	## TODO: Automate the process of getting these inputs. Make the inputs configuration parameters.

	def get_anchor_centers(self, im_height, im_width, sub_sample):
	
		'''
		Given image dimensions and subsampling ratio from backbone, we get all anchor centers with respect to original image.
		
		Inputs: 
		im_height: image height
		im_width: image width
		sub_sample: subsampling ratio of convolutional feature extractor
		
		Outputs:
		ctr_y, ctr_x: center points
		
		Testing the function:
		ctr_y, ctr_x = get_anchor_centers(im_height = 800, im_width = 450, sub_sample = 16)
		
		'''
		## Here x is facing towards the right, y is facing downwards
		num_of_y_pixels = im_height//sub_sample[0]
		num_of_x_pixels = im_width//sub_sample[1]

		## let's get anchor centers!
		ctr_x = np.arange(sub_sample[0]/2, (num_of_x_pixels)*sub_sample[0], sub_sample[0])
		ctr_y = np.arange(sub_sample[1]/2, (num_of_y_pixels)*sub_sample[1], sub_sample[1])

		return ctr_y, ctr_x

	def get_all_anchors(self,im_height, im_width, sub_sample, aspect_ratios, anchor_scales):
	
		'''
		Inputs: 
		
		im_height: image height
		im_width: image width
		sub_sample: subsampling ratio of convolutional feature extractor
		aspect_ratios: [ar1, ar2, ar3]
		anchor_scales: [as1, as2, as3]
		
		Output: 
		A numpy array
		
		'''
		
		## getting anchor centers with respect to original image
		ctr_y, ctr_x = self.get_anchor_centers(im_height=im_height, im_width = im_width, sub_sample = sub_sample)
		
		## At each pixel, we will have 9 anchors, total number of pixels in convolutional 
		## feature map are (im_height*im_width)//(sub_sample*sub_sample). Hence,
		## total number of anchors are,
		num_of_pixels = (im_height//sub_sample[0])*(im_width//sub_sample[1])
		anchors_per_pixel = len(aspect_ratios)*len(anchor_scales)
		
		## Each anchor has 4 points(y1, x1, y2, x2)
		## This variable holds information of all the anchors at all the pixel locations
		## defined in the original image
		anchor_data = np.zeros((num_of_pixels, anchors_per_pixel, 4), dtype=np.float32)
		
		
		## Let's iterate through all anchor centers and get their corresponding anchor co-ordinates
		index = 0
		for i in range(len(ctr_y)):
			for j in range(len(ctr_x)):
				
				anchor_center = [ctr_y[i], ctr_x[j]]
				
				## generate all 9 anchors for this anchor center, output 9x4 array
				anchor_base = self.generate_anchors(anchor_center, aspect_ratios, anchor_scales, sub_sample)
				
				## Populating anchor data matrix with appropriate co-ordinates
				anchor_data[index,:,:] = anchor_base
				
				index = index + 1
		
		anchor_data = anchor_data.reshape(num_of_pixels*anchors_per_pixel, 4)
		
		return anchor_data  ## Shape: Total number of anchors x 4