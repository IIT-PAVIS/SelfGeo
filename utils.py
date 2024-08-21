"""
This file contains functions used in SelfGeo

ZohaibMohammad.github.io
August 2024
"""


import torch
import torch.nn.functional as F
import numpy as np
import scipy.spatial.distance as distance


def compare_performance_deformable(kp, recons_shape, data, threshold=0.02, split='test',isreal=False):
    device = kp.device
    coverage = volume_loss(kp, data[0].float().to(device), split=split)
    inclusivity = surface_loss(kp, data[0].float().to(device), threshold=threshold, split=split)
    inclusivity_tau = torch.Tensor(surface_loss_diff_tau(kp, data[0].float().to(device), split='test')).cpu().tolist()
    
#    if isreal:
#        reconstruction_error = chamfer_distance_oneside(recons_shape, data[0].float().to(device))
#    else:
#        reconstruction_error = chamfer_distance(recons_shape, data[0].float().to(device))


    reconstruction_error = chamfer_distance(recons_shape, data[0].float().to(device))

    return coverage, inclusivity, reconstruction_error, inclusivity_tau


def compute_loss(kpts, pcds, wgd, recons_shape, writer, step, cfg, split='split??'):
    l_cov = cfg.parameters.cov * coverage_loss(kpts)     # separation in 3D space
    l_surf = cfg.parameters.surf * surface_loss(kpts, pcds)		# surface loss

    ''' Reconstructions losses '''
    if cfg.parameters.rec != 0:
        rec_loss = cfg.parameters.rec * chamfer_distance(pcds, recons_shape)
    else:
        rec_loss = 0.

    ''' Calculate geodesic loss among all the frames'''
    if cfg.parameters.geo != 0:
        geo_loss = 0
        wgd1 = wgd
        for i in range(wgd1.shape[0]):
            wgd1 = torch.roll(wgd1, shifts=1, dims=0)
            geo_loss += F.mse_loss(wgd1, wgd)
        geo_loss = geo_loss * cfg.parameters.geo
    else:
        geo_loss = 0.

    # -------------------------------------
    ''' semantic consistency between two consecutive frames '''
    kpts1 = kpts[0:-1, :, :]
    kpts2 = kpts[1:, :, :]
    smoothing_loss = cfg.parameters.smt * torch.mean(torch.norm(kpts1-kpts2, dim=-1))

    # -------------------------------------
    writer.add_scalar('{}_loss/chamf_loss1'.format(split), rec_loss, step)
    writer.add_scalar('{}_loss/separation'.format(split), l_cov, step)
    writer.add_scalar('{}_loss/shape'.format(split), l_surf, step)
    writer.add_scalar('{}_loss/geo_loss'.format(split), geo_loss, step)
    writer.add_scalar('{}_loss/smoothing_loss'.format(split), smoothing_loss, step)

    return rec_loss + l_cov + l_surf + smoothing_loss + geo_loss  # + dist_mat_loss1 +  frob_loss1 + l_volume +


def chamfer_distance(recons_pc, pc):
    '''
    Parameters
    ----------
    pc              Input point cloud
    recons_pc       Reconstructed point cloud

    Returns Shape loss -> how far the reconstructed points (PC) are estimated from the input point cloud
    -------

    '''
    pred_to_gt = torch.cat([torch.squeeze(
        torch.norm(pc[i].unsqueeze(1) - recons_pc[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(recons_pc))], dim=0)
    gt_to_pred = torch.cat([torch.squeeze(
        torch.norm(recons_pc[i].unsqueeze(1) - pc[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(pc))], dim=0)

    return (torch.mean(pred_to_gt) + torch.mean(gt_to_pred))/2


def chamfer_distance_oneside(pc, recons_pc):
    '''
    Parameters
    ----------
    pc              Input point cloud
    recons_pc       Reconstructed point cloud

    Returns Shape loss -> how far the reconstructed points (PC) are estimated from the input point cloud
    -------

    '''

    gt_to_pred = torch.cat([torch.squeeze(torch.norm(recons_pc[i].unsqueeze(1) - pc[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[0]) for i in range(len(pc))], dim=0)

    return torch.mean(gt_to_pred)


def surface_loss(kp, pc, threshold=0.15, split='train'):
    '''
    Parameters
    ----------
    pc      Input point cloud
    kp      Estimated key-points

    Returns Shape loss -> how far the key-points are estimated from the input point cloud
    -------

    '''
    loss = torch.cat([torch.squeeze(
        torch.norm(pc[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(kp))], dim=0)

    # pdb.set_trace()
    if split == 'test':
        return torch.tensor((len(loss[loss < threshold]) / len(loss)) * 100)  # percentage of points closest to the surface

    return torch.mean(loss)


def surface_loss_diff_tau(kp, pc, split='train'):
    '''
    Parameters
    ----------
    pc      Input point cloud
    kp      Estimated key-points
    device  cuda device name

    Returns Shape loss -> how far the key-points are estimated from the input point cloud
    -------

    '''

    tau_list = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    out = []

    loss = torch.cat([torch.squeeze(
        torch.norm(pc[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(kp))], dim=0)

    # pdb.set_trace()
    if split == 'test':
        for tau in tau_list:
            out.append(torch.tensor((len(loss[loss < tau]) / len(loss)) * 100))
        return out  # percentage of points closest to the surface

    return torch.mean(loss)



def coverage_loss(kp):
    '''
    Parameters
    ----------
    kp:         Key-points
    Method:     compute distances of each point from all the points in "kp"
                consider minimum two distances (distance of a point form itself (distance==0) and the next closest (distance>0))
                take mean of the distances from the closest point (distance>0)

    Returns     Coverage loss ->  average distance of every point from the closest points
    -------
    '''
    min_distances = torch.cat([torch.squeeze(
        torch.norm(kp[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None).topk(2, largest=False, dim=0)[
            0]) for i in range(len(kp))], dim=0)

    return 1/torch.mean(min_distances[min_distances>0])




def volume_loss(kp, pc, split='train'):
    '''

    Parameters: 3D Coverage loss
                => same as coverage loss of clara's Paper
                => https://github.com/cfernandezlab/Category-Specific-Keypoints/blob/master/models/losses.py
    Smooth L1 loss: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    ----------
    kp: Estimated key-points [BxNx3]
    pc: Point cloud [Bx2048x3]

    Returns: Int value -> IoU b/w kp and pc
    -------

    '''
    device = kp.device
    val_max_pc, _ = torch.max(pc, 1)    # Bx3
    val_min_pc, _ = torch.min(pc, 1)    # Bx3
    dim_pc = val_max_pc - val_min_pc    # Bx3
    val_max_kp, _ = torch.max(kp, 1)    # Bx3
    val_min_kp, _ = torch.min(kp, 1)    # Bx3
    dim_kp = val_max_kp - val_min_kp    # Bx3

    ''' % coverage of kp over pc'''
    if split == "test":
        temp = torch.tensor([[0,0,0]], dtype=torch.float32).to(device)
        dis_kp = torch.cdist(temp, dim_kp).squeeze() # distance of kp (BB) from origin
        dis_pc = torch.cdist(temp, dim_pc).squeeze() # distance of PC (BB) from origin

        overlapping = 1 - torch.abs(dis_pc - dis_kp) / dis_pc
        overlapping[overlapping<0] = 0
        return torch.mean(overlapping)*100 # percentage value

    return F.smooth_l1_loss(dim_kp, dim_pc)



def normalize_pc(pc):
	pc = pc - pc.mean(0)
	pc /= np.max(np.linalg.norm(pc, axis=-1)) # -1 to 1
	return pc/2  # # -0.5 to 0.5 (Unit box)


def normalize_pcd_kp(pc, kp, gt=None):
    mean_ = pc.mean(0)
    max_ = np.max(np.linalg.norm(pc, axis=-1))  # -1 to 1

    pc = pc - mean_
    pc /= max_  # -1 to 1

    kp = kp - mean_
    kp /= max_  # -1 to 1

    if gt != None:
        gt = gt - mean_
        gt /= max_  # -1 to 1

        return pc / 2, kp / 2, gt / 2  # # -0.5 to 0.5 (Unit box)

    return pc / 2, kp / 2  # # -0.5 to 0.5 (Unit box)


def knn(x, X, k, **kwargs):
    """
    find indices of k-nearest neighbors of x in X
    """
    d = distance.cdist(x.reshape(1,-1), X, **kwargs).flatten()
    return np.argpartition(d, k)[:k]


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
