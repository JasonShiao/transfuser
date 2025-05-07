import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import GlobalConfig
from model import LidarCenterNet
from data_latent import CARLA_Data2

import pathlib
import datetime
from torch.distributed.elastic.multiprocessing.errors import record
import random
import torch.multiprocessing as mp

from collections import deque
from copy import deepcopy
from data import lidar_to_histogram_features
from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # remove module.
        new_state_dict[new_key] = v
    return new_state_dict

# Records error and tracebacks in case of failure
@record
def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='transfuser', help='Unique experiment identifier.')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--load_file', type=str, default=None, help='ckpt to load.')
    parser.add_argument('--setting', type=str, default='all', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')
    parser.add_argument('--root_dir', type=str, default=r'/mnt/qb/geiger/kchitta31/datasets/carla/pami_v1_dataset_23_11', help='Root directory of your training data')
    parser.add_argument('--backbone', type=str, default='transFuser',
                        help='Which Fusion backbone to use. Options: transFuser, late_fusion, latentTF, geometric_fusion')
    parser.add_argument('--image_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the image branch. efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032',
                        help='Which architecture to use for the lidar branch. Tested: efficientnet_b0, resnet34, regnety_032 etc.')
    parser.add_argument('--use_velocity', type=int, default=0,
                        help='Whether to use the velocity input. Currently only works with the TransFuser backbone. Expected values are 0:False, 1:True')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers used in the transfuser')
    parser.add_argument('--use_target_point_image', type=int, default=1,
                        help='Valid values are 0, 1. 1 = using target point in the LiDAR0; 0 = dont do it')
    parser.add_argument('--use_point_pillars', type=int, default=0,
                        help='Whether to use the point_pillar lidar encoder instead of voxelization. 0:False, 1:True')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    shared_dict = None

    rank       = 0
    local_rank = 0
    world_size = 1
    device = torch.device('cuda:{}'.format(local_rank))

    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True # Wen want the highest performance

    # Configure config
    config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
    # Override config based on args
    config.use_target_point_image = bool(args.use_target_point_image)
    config.n_layer = args.n_layer
    config.use_point_pillars = bool(args.use_point_pillars)
    config.backbone = args.backbone

    # Create model (contains backbone + multiple heads) and optimizers
    model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, bool(args.use_velocity))
    model.cuda(device=device)

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr) # For single GPU training

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total trainable parameters (of model): ', params)

    # Data
    dataset = CARLA_Data2(root=config.train_data, config=config, shared_dict=shared_dict)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)

    # Create logdir
    if ((not os.path.isdir(args.logdir)) and (rank == 0)):
        print('Created dir:', args.logdir, rank)
        os.makedirs(args.logdir, exist_ok=True)

    # We only need one process to log the losses
    #writer = SummaryWriter(log_dir=args.logdir)
    # Log args
    # with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    if (not (args.load_file is None)):
        # Load checkpoint
        print("=============load=================")
        state_dict = torch.load(args.load_file, map_location=model.device)
        cleaned_state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(cleaned_state_dict, strict=False)
        #model.load_state_dict(torch.load(args.load_file, map_location=model.device))
        # optimizer.load_state_dict(torch.load(args.load_file.replace("model_", "optimizer_"), map_location=model.device))

    new_dataset = [] # (prev_latent, current_latent, next_latent)
    data_deque = deque(maxlen=3)
    for data in tqdm(dataloader):
        route_id, latent = load_data_compute_latent(model, data, device, config) # various loss terms
        print("latent shape:", latent.shape)
        # save latent
        latent = latent.cpu().detach().numpy()
        squeeze_latent = np.squeeze(latent)
        print("squeeze_latent shape:", squeeze_latent.shape)
        # 0. if route_id is different, clear
        # 1. append right, 
        # 2. update, 
        # 3. popleft
        if len(data_deque) != 0:
            if data_deque[0]['route_id'] != route_id:
                data_deque.clear()
                print('------ Changed route_id ------')
          
        data_deque.append({
          'route_id': route_id, 
          'latent': squeeze_latent
          }
        )
        if len(data_deque) == 3:
            # Update (Add to) dataset
            new_dataset.append({
              'prev': data_deque[0]['latent'], 
              'current': data_deque[1]['latent'], 
              'next': data_deque[2]['latent']}
            )
            #print(f"route_id: {data_deque[0]['route_id']}")
            #print(f"new data: {new_dataset[-1]['prev'][:10]}, {new_dataset[-1]['current'][:10]}, {new_dataset[-1]['next'][:10]}")
            data_deque.popleft()
            # save_path = os.path.join
    save_path = os.path.join("/home/ubuntu/transfuser/latent_pred_dataset", "latent_pred_dataset.npy")
    # save the latent dataset
    np.save(save_path, new_dataset)


# We need to seed the workers individually otherwise random processes in the dataloader return the same values across workers!
def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data_compute_latent(model, data, device, config):
    # Move data to GPU
    rgb = data['rgb'].to(device, dtype=torch.float32)
    depth = None
    semantic = None

    bev = data['bev'].to(device, dtype=torch.long)

    if (config.use_point_pillars == True):
        lidar = data['lidar_raw'].to(device, dtype=torch.float32)
        num_points = data['num_points'].to(device, dtype=torch.int32)
    else:
        lidar = data['lidar'].to(device, dtype=torch.float32)
        num_points = None
    # print("lidar shape:", lidar.shape)
    
    if (config.use_point_pillars == True):
        lidar_cloud = deepcopy(data['lidar'][1]) # exclude [0]: timestamp
        lidar_cloud[:, 1] *= -1  # invert
        lidar_bev = [torch.tensor(lidar_cloud).to('cuda', dtype=torch.float32)]
        num_points = [torch.tensor(len(lidar_cloud)).to('cuda', dtype=torch.int32)]
    else:
        lidar_transformed = deepcopy(data['lidar']) 
        lidar_transformed[:, 1] *= -1  # invert
        lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed)).unsqueeze(0)
        lidar_transformed_degrees = [lidar_transformed.to('cuda', dtype=torch.float32)]
        lidar_bev = torch.cat(lidar_transformed_degrees[::-1], dim=1)
        # print("lidar_bev shape:", lidar_bev.shape)

    # label = data['label'].to(self.device, dtype=torch.float32)
    # ego_waypoint = data['ego_waypoint'].to(self.device, dtype=torch.float32)
    target_point = data['target_point'].to(device, dtype=torch.float32)
    target_point_image = data['target_point_image'].to(device, dtype=torch.float32)
    ego_vel = data['speed'].to(device, dtype=torch.float32)

    features, image_features_grid, fused_latent_features = model.forward_backbone(rgb, lidar, 
                                                                                  target_point=target_point, target_point_image=target_point_image, 
                                                                                  ego_vel=ego_vel.reshape(-1, 1))
    # if ((self.args.backbone == 'transFuser') or (self.args.backbone == 'late_fusion') or (self.args.backbone == 'latentTF')):
    # elif (self.args.backbone == 'geometric_fusion'):
    #     raise Exception("Geometric fusion is not supported yet.")
    # else:
    #     raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")
    return data['route_id'], fused_latent_features

if __name__ == "__main__":
    # The default method fork can run into deadlocks.
    # To use the dataloader with multiple workers forkserver or spawn should be used.
    mp.set_start_method('fork')
    print("Start method of multiprocessing:", mp.get_start_method())
    main()
