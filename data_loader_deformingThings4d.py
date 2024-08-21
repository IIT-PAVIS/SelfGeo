import numpy as np
from glob import glob
import os
import pdb
import hydra
import torch
import omegaconf
from tqdm import tqdm
from natsort import natsorted
import open3d as o3d


BASEDIR = os.path.dirname(os.path.abspath(__file__))



def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


def add_noise(data, mu=0, sigma=0.05, size=0.05):

    # mu, sigma = 0, 4
    noise = np.random.normal(loc=mu, scale=sigma, size=data.shape) # [100])
    # noisy_data = data + noise
    # noise = np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)
    return data + noise


def uniform_sampling(data, npoints):
    '''

    Parameters
    ----------
    data   [B, N, 3]   =>  [8,2048,3]
    npoints number of required sample points

    Returns [B, npoints, 3]
    -------

    '''
    indices = np.linspace(0, data.shape[1]-1, npoints, dtype=np.int)  # 0, 2048, required_points
    return np.concatenate([np.expand_dims(data[:, i, :], 1) for i in indices], 1)


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1)) # -1 to 1
    return pc/2  # # -0.5 to 0.5 (Unit box)


def read_point_cloud(path):
    pc = o3d.io.read_point_cloud(path)
    return np.array(pc.points, np.float32)

def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
    return pc[idx[:n]]


class load_dataset(torch.utils.data.Dataset):
    '''
        Load the Deforming Things  4D dataset : PCDs and Geodesics
    '''
    def __init__(self, cfg, split):
        super().__init__()
        self.cfg = cfg
        class_name = cfg.class_name

        '''Read splits files and generate the paths '''
        split_models = open(os.path.join(BASEDIR, cfg.data.pcd_root,'../splits/{}_{}.txt'.format(class_name, split))).readlines()
        split_models = [m.rstrip('\n') for m in split_models]

        model_ids = []
        pcds = []
        geodesics_paths = []
        for model in split_models:
            paths = natsorted(glob(os.path.join(BASEDIR, self.cfg.data.pcd_root, 'pcds/{}/*.npy'.format(model))))
            for path in paths:
                file_name = path.split('/')[-1]
                model_ids.append(file_name.split('.')[0])
                pcds.append(np.load(path))
                geodesics_paths.append(os.path.join(BASEDIR, self.cfg.data.pcd_root, 'geodesics/{}/{}'.format(model, file_name)))
                
        self.model_ids = model_ids
        self.pcds = pcds
        self.geodesics_paths = geodesics_paths
        self.total_samples = len(self.pcds)
        print("\n\ntotal_samples: {}".format(self.total_samples))
        print("model_ids: {}".format(len(self.model_ids)))
        print("point clouds: {}".format(len(self.pcds)))
        print("geodesics: {}".format(len(self.geodesics_paths)))

    def __getitem__(self, idx):
        name = self.model_ids[idx]
        pcd = self.pcds[idx]
        geodesics = np.load(self.geodesics_paths[idx])

        if self.cfg.augmentation.normalize_pc:
            pcd = normalize_pc(pcd)

        if self.cfg.augmentation.uniform_sampling:
            pcd = farthest_point_sample(pcd, self.cfg.augmentation.sample_points)
            pcd = farthest_point_sample(pcd, 2048)

        if self.cfg.augmentation.gaussian_noise:
            pcd = add_noise(pcd, sigma=self.cfg.augmentation.lamda)

        return pcd, geodesics, name

    def __len__(self):
        return len(self.pcds)


class deformingThings_pck(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.cfg = cfg
        
        class_name = cfg.class_name
        sequences = natsorted(glob(os.path.join(BASEDIR, cfg.data.pcd_root_pck, class_name, '*.npz')))
        
        pcd0 = []
        offsets = []
        model_ids = []
        for seq in sequences:
            file = np.load(seq)
            pcd0.append(file['pcd0'])
            offsets.append(file['offsets'])
            file_name = seq.split('/')[-1]
            model_ids.append(file_name.split('.')[0])

        self.model_ids = model_ids
        self.pcds = pcd0
        self.offsets = offsets
        self.total_samples = len(self.pcds)
        print("\n\ntotal_samples: {}".format(self.total_samples))
        print("model_ids: {}".format(len(self.model_ids)))
        print("point clouds: {}".format(len(self.pcds)))
        print("offsets: {}".format(len(self.offsets)))

    def __getitem__(self, idx):
        name = self.model_ids[idx]
        pcd = self.pcds[idx]
        offsets = self.offsets[idx]

        ''' offsets are distances so normalization may change the shape '''

        return pcd, offsets, name

    def __len__(self):
        return len(self.pcds)


# main to test dataloader pipeline
def test_data_loader(cfg):

    train_dataset = load_dataset(cfg, 'test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers, drop_last=True)

    train_iter = tqdm(train_dataloader)

    for i, data in enumerate(train_iter):
        pdb.set_trace()
        pcd, name, geodesics  = data
        pcd1 = pcd[0:-1, :, :]
        pcd2 = pcd[1:, :, :]
        name1 = name[0:-1]
        name2 = name[1:]
        gd1 = geodesics[0:-1, :, :]
        gd2 = geodesics[1:, :, :]

        print(len(data[1]))
        print(data[0].shape)


@hydra.main(config_path='config', config_name='config_deforming_Things')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    # cfg.network.name = 'deformable_kp'
    # cfg.log_path = '{}_loader'.format(cfg.split)
    # logger.info(cfg.pretty())
    test_data_loader(cfg)

if __name__ == '__main__':
    main()
