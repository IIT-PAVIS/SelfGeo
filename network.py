import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import numpy as np
import open3d as o3d
import seaborn as sns

from networks.pointnetpp.pointnet2_sem_seg_msg import get_model as PointNetPP

class reconstuction_block(nn.Module):
    """
    # Residual block:
    # Input:    in_ (input channels)        # 1024
                out_ (output channels)      # 512
    # Output:
                x = [B x 512 x 2048]
    """
    def __init__(self, in_=10):
        super(reconstuction_block, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 2048, 1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, input):                       # [B x 10 x 3]
        x = F.relu(self.bn1(self.conv1(input)))     # [B x 256 x 3]
        x = F.relu(self.bn2(self.conv2(x)))         # [B x 512 x 3]
        x = F.relu(self.bn3(self.conv3(x)))         # [B x 1024 x 3]
        x = self.conv4(x)                           # [B x 2048 x 3]

        return x


class SelfGeo(nn.Module):
    """
        Unsupervised Key-point net: 3D key-points estimation from point clouds using an unsupervised approach
        Inputs:
            point-cloud: [Bx2048x3]
        Computes:
            - Point cloud features MxN [N (1024) features for every point (total N=2048)]
            - Reconstruction Block to reconstruct the shape of size Nx3
            - Conv1D computes N features for K key-points [KxN]
            - Soft-max normalize the features such that the sum of all the features (N) for single key-point (K1) become 1
              [KxN{0to1}]
            - Matrix Multiplication estimates the K key-points by averaging the points of the input PC based on the computed features
              So, the total [Kx3] key-points will be separated => that are the estimated key-points
        Output:
            - the key-points 3D positions [BxKx3]

        Sub-modules;
        1. Pointnet features extracter for point clouds
        2. Residual block, Conv1D ans Softmax
    """
    def __init__(self, cfg):
        super(SelfGeo, self).__init__()
        self.cfg = cfg
        self.keypoints = PointNetPP(cfg.key_points)
        self.reconstruct_shape = reconstuction_block(cfg.key_points)
        self.conv23 = torch.nn.Conv1d(cfg.key_points, 3, 1)

    def forward(self, pc, gd=None, name=""):
        APP_PT = torch.cat([pc, pc, pc], -1)  # input:[batch, n, 3]
        kp, w, f = self.keypoints(APP_PT.permute(0, 2, 1))  # PTW:[batch, 3*3, n] -> KP:[B, k, 3],  x: [B, k, 2048] & f1:[B, k, emb_dim(1024)]

        ''' Reconstruct the complete shape '''
        recons_shape = self.reconstruct_shape(kp)

        '''Weighted chamfer distance'''
        if gd == None:
            return kp, "", recons_shape  # [B, k, 3] ,[B, k, 3],  [B x 2048 x 3], [B x 2048 x 3]
        else:
            gd_kp = w @ gd @ torch.transpose(w,1,2)  # [B, k, 2048] * [B, 2048, 2048] => [B,k,2048] GD of the k keypoints w.r.t. other points in PCD
            return kp, gd_kp, recons_shape  # [B, k, 3] ,[B, k, 3],  [B x 2048 x 3], [B x 2048 x 3]
            

def show_points(points, kp=0, both=False):
    '''

    Parameters
    ----------
    points      point cloud  [2048, 3]
    kp          estimated key-points  [10, 3]
    both        if plot both or just the point clouds

    Returns     show the key-points/point cloud
    -------

    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    palette = sns.color_palette("bright", 25)  # create color palette
    if both==False:
        o3d.visualization.draw([pcd])
    else:
        key_points = []
        for i in kp:
            p = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            p.translate(i)
            p.paint_uniform_color(palette[2])
            key_points.append(p)

        o3d.visualization.draw_geometries([pcd, *key_points])

def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
    return pc[idx[:n]]


@hydra.main(config_path='config', config_name='config_cape')
def main(cfg):
    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    pc = np.load(BASEDIR + '/../../../dataset/CAPE_00032/test/pcd/shortlong_ballerina_spin.000009.npy')
    pc = torch.tensor(random_sample(pc,2048), dtype=torch.double).unsqueeze(0)
    data = torch.cat([pc,pc,pc], dim=0).float()
    # pc = torch.randn(5, 2048, 3)

    model = SelfGeo(cfg)
    kp,_,_ = model(data, None, "")
    print(kp.shape)
    show_points(data[0].numpy(), kp[0].detach().numpy(), True)

if __name__ == '__main__':
    main()


