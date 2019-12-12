
import torch
import numpy as np
import time
from ..utils import utils
import os
import sys
from ..eval.detection_map import DetectionMAP
import matplotlib.pyplot as plt

def test(model, data_loader, device, results_dir):
	is_training= False
	mAP = DetectionMAP(7)
	with torch.no_grad():
		for idx, batch_sample in enumerate(data_loader):

			in_images = batch_sample['image'].to(device)
			targets = [x.to(device) for x in batch_sample['target']]

			img_paths = batch_sample['image_path']

			# start = time.time()
			rpn_proposals, instances, proposal_losses, detector_losses = model(in_images, targets, is_training)
			# print(time.time() - start)
			utils.disk_logger(in_images, results_dir, rpn_proposals, instances, img_paths)
			# utils.ground_projection(in_images, results_dir, instances, img_paths)

			for instance, target in zip(instances, targets):
				pred_bb1 = instance.pred_boxes.tensor.cpu().numpy()
				pred_cls1 = instance.pred_classes.cpu().numpy() 
				pred_conf1 = instance.scores.cpu().numpy()
				gt_bb1 = target.gt_boxes.tensor.cpu().numpy()
				gt_cls1 = target.gt_classes.cpu().numpy()
				mAP.evaluate(pred_bb1, pred_cls1, pred_conf1, gt_bb1, gt_cls1)

	mAP.plot()
	plt.show()



def train(model, train_loader, val_loader, optimizer, epochs, tb_writer, lr_scheduler, device, model_save_dir, cfg):
	epoch = 1
	is_training = True

	while epoch <= epochs:

		running_loss = {}
		print("Epoch | Iteration | Loss ")

		for idx, batch_sample in enumerate(train_loader):

			in_images = batch_sample['image'].to(device)
			target = [x.to(device) for x in batch_sample['target']]

			rpn_proposals, instances, proposal_losses, detector_losses = model(in_images, target, is_training)

			loss_dict = {}
			loss_dict.update(proposal_losses)
			loss_dict.update(detector_losses) 

			loss = 0.0 
			for k, v in loss_dict.items():
				loss += v
			loss_dict.update({'tot_loss':loss})

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (idx)%10==0:
				# print(idx)
				# os.system('free -m')
				with torch.no_grad():
					#----------- Logging and Printing ----------#
					print("{:<8d} {:<9d} {:<7.4f}".format(epoch, idx, loss.item()))

					for loss_name, value in loss_dict.items():	
						tb_writer.add_scalar('Loss/'+loss_name, value.item(), epoch+0.01*idx)
					

					for key, value in loss_dict.items():
						if len(running_loss)<len(loss_dict):
							running_loss[key] = 0.0
						running_loss[key] = 0.9*running_loss[key] + 0.1*loss_dict[key].item()

					
					# utils.tb_logger(in_images, tb_writer, rpn_proposals, instances, "Training")
				#------------------------------------------------#
			# sys.exit()
		val_loss = {}
		# val_loss_error = []

		with torch.no_grad():
			for idx, batch_sample in enumerate(val_loader):
				
				in_images = batch_sample['image'].to(device)
				target = [x.to(device) for x in batch_sample['target']]

				rpn_proposals, instances, proposal_losses, detector_losses = model(in_images, target, is_training)

				loss_dict = {}
				loss_dict.update(proposal_losses)
				loss_dict.update(detector_losses)

				loss = 0.0 
				for k, v in loss_dict.items():
					loss += v
				loss_dict.update({'tot_loss':loss})

				for key, value in loss_dict.items():
					if len(val_loss)<len(loss_dict):
						val_loss[key] = []
					val_loss[key].append(loss_dict[key].item())

				# utils.tb_logger(in_images, tb_writer, rpn_proposals, instances, "Validation")

			for key, value in  val_loss.items():
				val_loss[key] = np.mean(val_loss[key])

			for key in val_loss.keys():
				tb_writer.add_scalars('loss/'+key, {'validation': val_loss[key], 'train': running_loss[key]}, epoch)


			print("Epoch ---- {} ".format(epoch))
			print("Epoch      Training   Validation")
			print("{} {:>13.4f}    {:0.4f}".format(epoch, running_loss['tot_loss'], val_loss['tot_loss']))
			
		## Decaying learning rate
		lr_scheduler.step()

		print("Epoch Complete: ", epoch)
		# # Saving at the end of the epoch
		if epoch % cfg.TRAIN.SAVE_MODEL_EPOCHS == 0:
			model_path = model_save_dir + "/epoch_" +  str(epoch).zfill(5) + '.model'
			torch.save({
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'lr_scheduler' : lr_scheduler.state_dict(),
					'cfg': cfg
					 }, model_path)

		epoch += 1

	tb_writer.close()