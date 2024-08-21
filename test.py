'''
SelfGeo: Self-supervised and Geodesic-consistent Estimation of Keypoints on Deformable Shapes
ECCV 2024


This file cann be used to test both CAPE and Deforming Things 4D dataset:

For CAPE dataset use the following two lines:
		 import data_loader_cape as dataset 
		 @hydra.main(config_path='config', config_name='config_cape') 
		 
For DeformingThings4D dataset use the following two lines:
		import data_loader_deformingThings4d as dataset
		@hydra.main(config_path='config', config_name='config_deforming_Things') 


In case of any query, feel free to contact.



Mohammad Zohaib
zohaib.mohammad@hotmail.com
zohaibmohammad.github.io


****************** Further details ****************************************


CAPE dataset:
    data loader: import data_loader_cape as dataset
    hydra_config: @hydra.main(config_path='config', config_name='config_cape')
    config file: config/config_cape.yaml
        class_name: human
        split: test
        pcd_root: dataset/CAPE_00032
        best_model_path: outputs/train/human/Best_model_human_12kp.pth

DeformingThings4D dataset:
    data loader: import data_loader_deformingThings4d as dataset
    hydra_config: @hydra.main(config_path='config', config_name='config_deforming_Things')
    config file: config/config_deforming_Things.yaml
        class_name: tigerD8H
        split: test
        pcd_root_pck: dataset/DeformingThings4D/pcds_geodesics  # pcds and geodesics
        best_model_path: 'outputs/train/tigerD8H/Best_model_tigerD8H_12kp.pth' # Please train the network to generate the  Best_model_human_12kp.pth


'''



import os
import hydra
import torch
import omegaconf
from tqdm import tqdm

# import data_loader_cape as dataset       # CAPE dataset
import data_loader_deformingThings4d as dataset   # Deforming things 4D dataset
import utils as function_bank
import visualizations as visualizer
import torch.nn.functional as F
import numpy as np
import network
import logging


logger = logging.getLogger(__name__)
BASEDIR = os.path.dirname(os.path.abspath(__file__))

def test(cfg):
    test_dataset = dataset.load_dataset(cfg, 'test')
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

    coverage = []
    inclusivity = []
    recons_error = []
    inclusivity_diff_tau = []
    consistency_list = []
    gd_error = []
    for epoch in range(1):
        model.eval()
        meter.reset()
        test_iter = tqdm(test_dataloader)
        for i, data in enumerate(test_iter):
            with torch.no_grad():
                pcd = data[0].float().to(device)
                kp, wgd, recons_shape = model(pcd, data[1].float().to(device), name=data[2][0])

                # comment to avoid visualizations
                if cfg.save_results:
                    visualizer.save_keypoints(data[0][0].cpu().numpy(), kp[0].cpu().numpy(), name=data[2][0])

                # Evaluations
                coverage_, inclusivity_, reconstruction_error_, inclusivity_tau_ = function_bank.compare_performance_deformable(kp, recons_shape, data, threshold = cfg.thresholds.inclusivity, split='test', isreal=False)

                coverage.append(coverage_.cpu())
                inclusivity.append(inclusivity_.cpu())
                recons_error.append(reconstruction_error_.cpu())
                inclusivity_diff_tau.append(inclusivity_tau_)
                
                if i==0:
                    previous_kp = kp[0]
                    previous_gd = wgd[0]
                else:
                    consistency_list.append(torch.argmin(torch.norm(kp[0].unsqueeze(1) - previous_kp.unsqueeze(0), dim=2, p=None), 1).cpu().tolist())
                    gd_error.append(F.mse_loss(wgd[0], previous_gd).cpu())
                    previous_kp = kp[0]
                    previous_gd = wgd[0]
                
        coverage = np.asarray(coverage)
        logger.info(f'Total frames: {len(coverage)}')
        logger.info(f'Avg: coverage : {np.mean(coverage)}')

        inclusivity = np.asarray(inclusivity)
        logger.info(f'Avg: inclusivity {np.mean(inclusivity)}')
        
        recons_error = np.asarray(recons_error)
        logger.info(f'Avg: recons_error {np.mean(recons_error)}')

        tau_list = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        logger.info(f'tau_list:  {tau_list}')

        inclusivity_diff_tau = np.asarray(inclusivity_diff_tau)
        logger.info(f'Avg inclusivity for different thresholds:')
        inclusivity_for_each_tau = [np.mean(inclusivity_diff_tau[:,i]) for i in range(inclusivity_diff_tau.shape[1])]
        logger.info(f'inclusivity_for_each_tau:  {inclusivity_for_each_tau}')

        temporal_consistency = []
        consistency_list = np.asarray(consistency_list)
        for j in range(consistency_list.shape[1]):
            temporal_consistency.append(np.sum(consistency_list[:, j] == j) / len(consistency_list)*100)
        logger.info(f'Temporal_consistency for each keypoint: {temporal_consistency}')
        logger.info(f'Average temporal_consistency for whole test set: {np.mean(temporal_consistency)}')
        
        gd_error = np.asarray(gd_error)
        logger.info(f'gd_error: {np.mean(gd_error)}')
        

# @hydra.main(config_path='config', config_name='config_cape')  # To test on CAPE dataset
@hydra.main(config_path='config', config_name='config_deforming_Things')   # To test on DeformingThings4D dataset

def main(cfg):
    cfg.split = 'test'  # we are in testing phase
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.log_path = '{}_logs'.format(cfg.task)
    logger.info(cfg.pretty())
    test(cfg)


if __name__ == '__main__':
    main()

