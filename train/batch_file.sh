#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --gres=gpu:1            # request GPU "generic resource"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1	# maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham
#SBATCH --mem=12000M		# memory per node
#SBATCH --time=00-24:00		# time (DD-HH:MM)
#SBATCH --output=%N-%j.out	# %N for node name, %j for jobID
#SBATCH --requeue
#SBATCH --mail-user=dhaivat1994@gmail.com
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
module load cuda
conda activate maskrcnn_benchmark
cd /home/dhai1729/Denso-OD/train/
# export TMPDIR="/home/dhai1729/scratch/MILA_cluster"

# export NGPUS=4
# ln -nsf $TMPDIR/coco_dataset_new/val2017_modified/ datasets/coco/val2017_new
# ln -nsf $TMPDIR/coco_dataset_new/train2017_modified/ datasets/coco/train2017_new
# ln -nsf $TMPDIR/coco_dataset_new/annotations_modified/instances_val2017_modified.json datasets/coco/annotations/instances_val2017_new.json
# ln -nsf $TMPDIR/coco_dataset_new/annotations_modified/instances_train2017_modified.json datasets/coco/annotations/instances_train2017_new.json
# export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$EBROOTCUDNN/lib64:$LD_LIBRARY_PATH
echo Running on $HOSTNAME

python train.py
