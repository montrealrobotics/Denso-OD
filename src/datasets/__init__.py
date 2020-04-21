from .kitti_dataloader import KittiDataset
from .kitti_mot_dataloader import KittiMOTDataset_KF, KittiMOTDataset

def build_dataset(data_name):
	all_datasets = {"KittiDataset": KittiDataset,
					"KittiMOTDataset":KittiMOTDataset,
					"KittiMOTDataset_KF": KittiMOTDataset_KF}
	return all_datasets[data_name]
