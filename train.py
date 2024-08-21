'''

SelfGeo: Self-supervised  and Geodesic-consistent Estimation of Keypoints on Deformable Shapes
ECCV 2024


This file can be used to train both CAPE and Deforming Things 4D dataset:

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
        split: train
        pcd_root: dataset/CAPE_00032
        best_model_path: 'path_to_best_weights'

DeformingThings4D dataset:
    data loader: import data_loader_deformingThings4d as dataset
    hydra_config: @hydra.main(config_path='config', config_name='config_deforming_Things')
    config file: config/config_deforming_Things.yaml
        class_name: tigerD8H
        split: train
        pcd_root: dataset/DeformingThings4D/pcds_geodesics  # pcds and geodesics
        best_model_path: 'path_to_best_weights'


'''



import os
import hydra
import torch
import omegaconf
from tqdm import tqdm

# import data_loader_cape as dataset       # CAPE dataset
import data_loader_deformingThings4d as dataset   # Deforming things 4D dataset

from utils import AverageMeter
import utils as function_bank
from torch.utils.tensorboard import SummaryWriter
import network
import logging
logger = logging.getLogger(__name__)


def train(cfg):
    writer = SummaryWriter("train_summary")

    train_dataset = dataset.load_dataset(cfg, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    val_dataset = dataset.load_dataset(cfg, 'val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.SelfGeo(cfg).to(device)
    best_model_path = cfg.data.best_model_path
    if os.path.isfile(best_model_path):
        logger.info('\n\n best_model_path : {}\n\n'.format(best_model_path))
        model.load_state_dict(torch.load(best_model_path))
    else:
        logger.info("\nBest model not found. Stating training from the beginning ... ")


    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    meter = AverageMeter()
    best_loss = 1e10
    train_step = 0
    val_step = 0
    for epoch in range(cfg.max_epoch):
        train_iter = tqdm(train_dataloader)

        # Training
        meter.reset()
        model.train()
        for i, data in enumerate(train_iter):
            pcd = data[0].float().to(device) 
            kp, wgd, recons_shape = model(pcd, data[1].float().to(device), name=data[2][0])
            loss = function_bank.compute_loss(kp, pcd, wgd, recons_shape, writer, train_step, cfg, split='train')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss/overall', loss, train_step)  # write training loss
            train_step += 1  # increment in train_step

            train_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())

        train_loss = meter.avg
        logger.info(f'Epoch: {epoch}, Average Train loss: {meter.avg}')

        # Validation
        model.eval()
        meter.reset()
        val_iter = tqdm(val_dataloader)
        for i, data in enumerate(val_iter):
            with torch.no_grad():
                pcd = data[0].float().to(device) 
                kp, wgd, recons_shape = model(pcd, data[1].float().to(device), name=data[2][0])
                loss = function_bank.compute_loss(kp, pcd, wgd, recons_shape, writer, val_step, cfg, split='val')

                writer.add_scalar('val_loss/overall', loss, val_step)  # write validation loss
                val_step += 1  # increment in val_step

            val_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())

        val_loss = meter.avg
        if meter.avg < best_loss:
            logger.info("best epoch: {}".format(epoch))
            best_loss = meter.avg
            torch.save(model.state_dict(),'Best_model_{}_{}kp.pth'.format(cfg.class_name, cfg.key_points))

        logger.info(f'Epoch: {epoch}, Average Val loss: {meter.avg}')
        writer.add_scalars('loss_per_epoch', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)  # write validation loss

    writer.close()  # close the summary writer
    logger.info(" Reached to {} epoch \n".format(cfg.max_epoch))


# @hydra.main(config_path='config', config_name='config_cape')  # To train on CAPE dataset
@hydra.main(config_path='config', config_name='config_deforming_Things')   # To train on DeformingThings4D dataset
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.log_path = '{}_logs'.format(cfg.task)
    logger.info(cfg.pretty())
    train(cfg)



if __name__ == '__main__':
    main()

