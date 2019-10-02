import torch, torchvision
import torchvision.transforms as T
from kitti_dataloader import NuScenesDataset

transform = T.Compose([T.ToTensor()])

def nuscenes_collate_fn(batch):
    
    ## let's get the images stacked up
    data = torch.stack([item[0] for item in batch])
    ## getting the target in a list
    target = [item[1] for item in batch]
    ## Getting paths in list
    paths = [item[2] for item in batch]
    # target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    return [data, target, paths]

kitti_dataset = NuScenesDataset("/home/dishank/projects/MILA/Denso-OD/kitti_dataset", transform = transform)
kitti_train_loader = torch.utils.data.DataLoader(kitti_dataset, batch_size=1, shuffle=True, collate_fn = nuscenes_collate_fn)


for images, target, path in kitti_train_loader:
    print(len(target[0]))


