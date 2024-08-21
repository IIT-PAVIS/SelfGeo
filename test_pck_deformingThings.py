'''
SelfGeo: Self-supervised and Geodesic-consistent Estimation of Keypoints on Deformable Shapes
ECCV 2024


- This file can be used to compute the PCK
- We only use the DeformingThings4D dataset for PCK computation
- First, generate the "correspondences_for_pck" folder by running the "sdk.py", otherwise, data will not be loaded
- set the path of 'pcd_root_pck' in the configure file (config_deforming_Things.yaml)

In case of any query, feel free to contact.

Mohammad Zohaib
zohaib.mohammad@hotmail.com
zohaibmohammad.github.io


****************** Further details ****************************************

DeformingThings4D dataset:
    data loader: import data_loader_deformingThings4d as dataset
    hydra_config: @hydra.main(config_path='config', config_name='config_deforming_Things')
    config file: config/config_deforming_Things.yaml
        class_name: tigerD8H
        split: test
        pcd_root_pck: dataset/DeformingThings4D/correspondences_for_pck  # pcds with frame-wise correspondences
        best_model_path: 'outputs/train/tigerD8H/Best_model_tigerD8H_12kp.pth' # Please train the network to generate the  Best_model_human_12kp.pth


'''


import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

import hydra
import torch
import omegaconf
from tqdm import tqdm
import data_loader_deformingThings4d as dataset
import utils as function_bank
import visualizations as visualizer
import numpy as np
import copy
import network
import logging

logger = logging.getLogger(__name__)
BASEDIR = os.path.dirname(os.path.abspath(__file__))


def pck_test(cfg):
    '''
    Load the Deforming Things 4D datasets
    Compute the reference keypoints on the first frame
    Use the available offsets to transform the reference keypoints to next frames
    compare the transformed reference keypoints with the estimated keypoints of the corresponding frame to compute the PCK

    Visualize the reference and estimated keypoints using the function: visualizer.save_two_keypoints
    Comment the function if you do not want to save/visualize the keypoints

    The PCK values will be displayed on the console window

    NOTE: the reference values are the distances between the corresponding points in two frames.
          therefore, we can not normalize the original dataset
          Instead, we use original dataset and normalize the keypoints and PCDs for PCK calculation

    For any query, pls contact: zohaib.mohammad@hotmail.com
    -------

    '''

    test_dataset = dataset.deformingThings_pck(cfg, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.SelfGeo(cfg).to(device)
    best_model_path = os.path.join(BASEDIR, cfg.data.best_model_path)
    if os.path.isfile(best_model_path):
        logger.info('\n\n best_model_path : {}\n\n'.format(best_model_path))
        model.load_state_dict(torch.load(best_model_path))
    else:
        logger.info("\nBest model not found. Stating training from the beginning ... ")
        return 0

    meter = function_bank.AverageMeter()
    pck = []    # 0.01, 0.01, 0.03, 0.04, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,0.10
    model.eval()
    meter.reset()
    test_iter = tqdm(test_dataloader)
    for i, data in enumerate(test_iter):
        with torch.no_grad():
            ''' Estimate keypoints from first frame '''
            pcd0 = copy.deepcopy(data[0].float())
            pcd0 = function_bank.normalize_pc(pcd0[0]).unsqueeze(0)
            kp0, _, _ = model(pcd0.to(device), None, name=data[2][0])
            kp0 = kp0[0].cpu()

            '''
               Create reference keypoints (points of the first PCD close to the estimated keypoints)
               The points of the PCD (keypoints on shape) can be transformed to next frames using offsets 
            '''
            ref_index0 = []
            for kp in kp0:
                ref_index0.append(function_bank.knn(kp, pcd0[0], 1))
            ref_index0 = np.asarray(ref_index0).squeeze()
            ref_kp0 = data[0][0][ref_index0]
            ref_kp0 = torch.tensor(ref_kp0)

            ''' Estimate keypoints on the next frames '''
            pck_per_obj = []
            for i, offset in enumerate(data[1][0]):
                pcdi = copy.deepcopy(data[0].float()+ offset.unsqueeze(0))  # Next frame
                pcdi_n = function_bank.normalize_pc(pcdi[0]).unsqueeze(0)

                kpi, _, _ = model(pcdi_n.to(device), None, name=data[2][0])     # Keypoints for next frame
                kpi = kpi[0].cpu()

                ref_kpi = []
                for kp in kpi:
                    ref_kpi.append(function_bank.knn(kp, pcdi_n[0], 1))
                ref_kpi = np.asarray(ref_kpi).squeeze()
                kpi_ = pcdi[0][ref_kpi]     # Estimated keypoints on PCDs

                gt_kp0 = ref_kp0+offset[ref_index0]     # Transformed keypoints of the first frame

                pcdi_pts, kpi_, gt_kp0 = function_bank.normalize_pcd_kp(pcdi[0], kpi_, gt_kp0) # normalize all

                ''' Save/visualize the estimated keypoints of next frame and trannnnsformed reference keypoints of the first frame '''
                if cfg.save_results:
                    visualizer.save_two_keypoints(pcdi_pts, kpi_, gt_kp0, folder="pcd_debug", name="{}_".format(i)+data[2][0], save=True)

                ''' Distance between the estimated and transformed reference keypoints '''
                kpi_ = torch.tensor(kpi_)
                eucli_dis_mat = torch.cdist(gt_kp0, kpi_, p=2)
                dis_corrs_kpts = []
                for i in range (len(kpi_)):
                    dis_corrs_kpts.append(eucli_dis_mat[i][i].item())
                dis_corrs_kpts = np.asarray(dis_corrs_kpts)

                ''' Compute PCK considering different tau values '''
                tau_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
                pck_per_frame = []
                for tau in tau_list:
                    pck_per_frame.append((len(dis_corrs_kpts[dis_corrs_kpts < tau]) / len(dis_corrs_kpts)) * 100)

                ''' Save the PCK for every frame '''
                pck_per_obj.append(pck_per_frame)
                pck.append(pck_per_frame)

            pck_per_obj = np.asarray(pck_per_obj)

            logger.info('\n category = {}'.format(data[2][0]))
            logger.info('pck_per_obj at tau 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] = ')
            logger.info(' {:.2f}, {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}, {:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(pck_per_obj[:,0]),
                                                                                                          np.mean(pck_per_obj[:,1]), np.mean(pck_per_obj[:,2]), np.mean(pck_per_obj[:,3]),
                                                                                                          np.mean(pck_per_obj[:,4]), np.mean(pck_per_obj[:,5]), np.mean(pck_per_obj[:,6]),
                                                                                                          np.mean(pck_per_obj[:,7]), np.mean(pck_per_obj[:,8]), np.mean(pck_per_obj[:,9])))

    pck = np.asarray(pck)

    logger.info('\n\n\n\n\n------------------------\n\n')
    logger.info('\n category = {}'.format(data[2][0]))
    logger.info('PCK for all testset for tau 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] = ')
    logger.info(' {:.2f}, {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}, {:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(pck[:,0]),
                                                                                                  np.mean(pck[:,1]), np.mean(pck[:,2]), np.mean(pck[:,3]),
                                                                                                  np.mean(pck[:,4]), np.mean(pck[:,5]), np.mean(pck[:,6]),
                                                                                                  np.mean(pck[:,7]), np.mean(pck[:,8]), np.mean(pck[:,9])))


@hydra.main(config_path='config', config_name='config_deforming_Things')
def main(cfg):
    cfg.split = 'test'
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.log_path = '{}_logs'.format(cfg.task)
    logger.info(cfg.pretty())

    pck_test(cfg)

if __name__ == '__main__':
    main()

