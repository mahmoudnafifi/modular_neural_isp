"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

This file contains the training script for the detail-enhancement model.
"""

import argparse
import logging
import os
import sys

import numpy as np
import tensorboard.summary
from tabulate import tabulate
import shutil

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
from utils.file_utils import write_json_file

from dataset import Data
from torch.optim.lr_scheduler import CosineAnnealingLR

from main.pipeline import PipeLine
from denoising.nafnet_arch import NAFNet
from utils.constants import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

def print_line(end: Optional[bool]=False, length: Optional[int]=100):
  """Prints a separator line."""
  line = '-' * length
  if end:
    print(f'\n{line}\n')
  else:
    print(f'\n{line}')

def training(model: NAFNet, epochs: int, lr: float, l2_reg: float, tr_device: torch.device,
             train_loader: DataLoader, val_loader: DataLoader, train: Data, global_step: List[int],
             validation_frequency: int, exp_name: str, batch_size: int, writer: tensorboard.summary.Writer, log: Dict):
  """Performs training on the given dataloaders."""

  optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2_reg)
  scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

  for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    with tqdm(total=len(train) * batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
      for batch_idx, batch in enumerate(train_loader):
        images = batch['in_images'].squeeze(0)
        gt_images = batch['gt_images'].squeeze(0)
        in_images = images.to(device=tr_device, non_blocking=True)
        gt_images = gt_images.to(device=tr_device, non_blocking=True)
        out_images = model(in_images)
        loss = F.l1_loss(gt_images, out_images)

        epoch_loss += loss.item()

        if writer:
          writer.add_scalar(f'Loss/train', loss.item(), global_step[0])
        pbar.set_postfix({f'Batch-loss': f'{loss.item()}'})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(np.ceil(images.shape[0]))
        global_step[0] += 1

    if (epoch + 1) % validation_frequency == 0:
      val_loss = validate(model=model, loader=val_loader, val_device=tr_device, writer=writer,
                          global_step=global_step[0])
      print_line()

      logging.info(f'Validation loss: {val_loss}\n')

      checkpoint_model_name = os.path.join('checkpoints', f'{exp_name}_{epoch + 1}.pth')
      torch.save(model.state_dict(), checkpoint_model_name)
      logging.info(f'Checkpoint {epoch + 1} saved!')
      print_line(end=True)

      log['checkpoint_model_name'].append(checkpoint_model_name)
      log['val_l1'].append(val_loss)
      write_json_file(log, os.path.join('logs', f'{exp_name}'))

      if writer:
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step[0])
        writer.add_scalar(f'Loss/val', val_loss, global_step[0])
        writer.add_images('Input images/train', images, global_step[0])
        writer.add_images('Output images/train', out_images, global_step[0])
        writer.add_images('GT images/train', gt_images, global_step[0])
    scheduler.step()

  torch.save(model.state_dict(), os.path.join('models', f'{exp_name}.pth'))
  logging.info('Saved trained model!')
  best_model_idx = log['val_l1'].index(min(log['val_l1']))
  best_model_name = log['checkpoint_model_name'][best_model_idx]
  shutil.copy(best_model_name, os.path.join('models', f'{exp_name}-best.pth'))

def train_net(model: NAFNet, tr_device: torch.device, in_tr_dir: str, gt_tr_dir: str,
              data_tr_dir: str, in_val_dir: str, gt_val_dir: str, data_val_dir: str, epochs: int, batch_size: int,
              lr: float, l2_reg: float, in_sz: int, validation_frequency: int, exp_name: str,
              temp_folder: str, overwrite_temp_folder: bool, delete_temp_folder: bool,
              no_tensorboard: bool, denoising_model_paths: List[str], temp_folder_postfix: str,
              pipeline_model: PipeLine, no_downsampling: bool):

  """Trains network."""

  print_line()
  print(f'Training on {in_sz}x{in_sz} images ...')
  print_line(end=True)
  writer = SummaryWriter(comment=f'TB-{exp_name}') if SummaryWriter and not no_tensorboard else None
  global_step = [0]

  train = Data(in_img_dir=in_tr_dir, gt_img_dir=gt_tr_dir, data_dir=data_tr_dir if data_tr_dir is None else data_tr_dir,
               temp_folder=temp_folder, overwrite_temp_folder=overwrite_temp_folder, batch_size=batch_size,
               image_size=in_sz, shuffle=True, geometric_aug=True, extract_patches=True,
               temp_folder_postfix=temp_folder_postfix,
               denoising_model_paths=denoising_model_paths, pipeline_model=pipeline_model,
               ps_downsampling=not no_downsampling)

  val = Data(in_img_dir=in_val_dir, gt_img_dir=gt_val_dir,
             data_dir=data_val_dir if data_val_dir is None else data_val_dir, image_size=in_sz,
             temp_folder=temp_folder, overwrite_temp_folder=overwrite_temp_folder, geometric_aug=False,
             batch_size=batch_size, shuffle=True, extract_patches=True,
             temp_folder_postfix=temp_folder_postfix,
             denoising_model_paths=denoising_model_paths, pipeline_model=pipeline_model,
             ps_downsampling=not no_downsampling)

  train_loader = DataLoader(train, batch_size=1, num_workers=16, pin_memory=True, persistent_workers=False,
                            shuffle=True)

  val_loader = DataLoader(val, batch_size=1, num_workers=16, pin_memory=True, drop_last=True,
                          persistent_workers=False, shuffle=True)

  log = {'checkpoint_model_name': [], 'val_l1': []}

  training(model=model, epochs=epochs, lr=lr, l2_reg=l2_reg, tr_device=tr_device,
           train_loader=train_loader, val_loader=val_loader, train=train, global_step=global_step,
           validation_frequency=validation_frequency, exp_name=exp_name, batch_size=batch_size, writer=writer, log=log)

  if writer:
    writer.close()
  logging.info(f'End of training.')

  if delete_temp_folder:
    postfix = '_patches_' + temp_folder_postfix
    logging.info('Deleting temp folders')
    tr_temp_dir = os.path.join(os.path.dirname(gt_tr_dir),
                               f'{temp_folder}_{os.path.basename(gt_tr_dir)}_bs_{batch_size}_sz_{in_sz}{postfix}')

    val_temp_dir = os.path.join(os.path.dirname(gt_val_dir),
                                f'{temp_folder}_{os.path.basename(gt_val_dir)}_bs_{batch_size}_sz_{in_sz}{postfix}')
    shutil.rmtree(tr_temp_dir)
    if tr_temp_dir != val_temp_dir:
      shutil.rmtree(val_temp_dir)
    logging.info('Done!')


def validate(model: PipeLine, loader: DataLoader, val_device: torch.device, writer: SummaryWriter, global_step: int,
             ) -> Dict[str, float]:
  """Network validation."""
  model.eval()

  valid_loss = 0
  with torch.no_grad():
    for idx, batch in enumerate(loader):
      in_images = batch['in_images'].squeeze(0)
      gt_images = batch['gt_images'].squeeze(0)
      in_images = in_images.to(device=val_device, non_blocking=True)
      gt_images = gt_images.to(device=val_device, non_blocking=True)
      out_images = model(in_images)
      loss = F.l1_loss(gt_images, out_images)
      valid_loss += loss.item()

  if writer:
    writer.add_images('Input images/val', in_images, global_step)
    writer.add_images('Output images/val', out_images, global_step)
    writer.add_images('GT images/val', gt_images, global_step)

  valid_loss /= idx
  model.train()
  return valid_loss


def get_args():
  parser = argparse.ArgumentParser(description='Pipeline tuning.')
  parser.add_argument(
    '--in-training-dir', dest='in_tr_dir', type=str, help='Training input image directory.', required=True)
  parser.add_argument(
    '--gt-training-dir', dest='gt_tr_dir', type=str, help='Training ground-truth image directory.',
    required=True)
  parser.add_argument(
    '--data-training-dir', dest='data_tr_dir', type=str, default=None,
    help='Training input data directory.')
  parser.add_argument(
    '--in-validation-dir', dest='in_vl_dir', type=str, help='Validation input image directory.',
    required=True)
  parser.add_argument(
    '--gt-validation-dir', dest='gt_vl_dir', type=str, help='Validation ground-truth image directory.',
    required=True)
  parser.add_argument('--data-validation-dir', dest='data_vl_dir', default=None, type=str,
                      help='Validation input data directory.')
  parser.add_argument('--epochs', type=int, default=50, dest='epochs')
  parser.add_argument('--batch-size', type=int, default=16, dest='batch_size')
  parser.add_argument('--learning-rate', type=float, default=1e-4, dest='lr')
  parser.add_argument('--l2reg', type=float, default=0.0000001, help='L2 Regularization factor',
                      dest='l2_r')
  parser.add_argument( '--load', dest='load', type=str, default=None,
                       help="Load enhancement model's weights from a .pth file")
  parser.add_argument( '--photofinishing-model-path', dest='photofinishing_model_path', type=str,
                       help='Path to trained Photofinishing .pth file', required=True)
  parser.add_argument( '--denoising-model-path', dest='denoising_model_path', type=str, nargs='+',
                       help='Path to trained denoising .pth file(s).', required=True)
  parser.add_argument('--validation-frequency', dest='val_frq', type=int, default=4,
                      help='Frequency (in epochs) to perform validation and save checkpoints.')
  parser.add_argument('--in-size', dest='in_sz', type=int,
                      default=ENHANCEMENT_TRAINING_INPUT_SIZE, help='Size of training patches.')
  parser.add_argument('--no-downsampling', dest='no_downsampling', action='store_true',
                      help='Disable image downsampling before photofinishing and skip guided upsampling after.')
  parser.add_argument('--temp-folder', dest='temp_folder', type=str, default='en_temp_h5',
                      help='Name of temporary folder to save training data.')
  parser.add_argument('--overwrite-temp-folder', dest='overwrite_temp_folder', action='store_true',
                      help='Overwrite "--temp-folder" if it exists.')
  parser.add_argument('--no-tensorboard', dest='no_tensorboard', action='store_true',
                      help='To skip TensorBoard logging.')
  parser.add_argument('--delete-temp-folder', dest='delete_temp_folder', action='store_true',
                      help='To delete "--temp-folder" after training.')
  parser.add_argument('--exp-name', type=str, default=None)
  return parser.parse_args()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  assert args.in_sz >= 256, 'Expected input size >= 256.'
  assert os.path.exists(args.photofinishing_model_path)

  print(tabulate([(key, value) for key, value in vars(args).items()], headers=['Argument', 'Value'], tablefmt='grid'))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info(f'Using device {device}')

  model_name = 'enhancement'
  if args.exp_name is not None and args.exp_name != '':
    model_name += f'_{args.exp_name}'

  os.makedirs('models', exist_ok=True)
  os.makedirs('configs', exist_ok=True)
  os.makedirs( 'checkpoints', exist_ok=True)
  os.makedirs('logs', exist_ok=True)

  config = {'width': ENHANCEMENT_MODEL_WIDTH,
            'middle_block_num': ENHANCEMENT_MODEL_MIDDLE,
            'encoder_block_nums': ENHANCEMENT_MODEL_ENCODER,
            'decoder_block_nums': ENHANCEMENT_MODEL_DECODER
            }
  config_file_name = model_name + '.json'

  logging.info(f'Tuning of raw enhancement module -- model name: {model_name} ...')

  write_json_file(config, os.path.join('configs', config_file_name))

  enhancement_model = NAFNet(width=config['width'], middle_block_num=config['middle_block_num'],
                                 encoder_block_nums=config['encoder_block_nums'],
                                 decoder_block_nums=config['decoder_block_nums']).to(device)

  print(f'Total number of enhancement network params: {sum(p.numel() for p in enhancement_model.parameters())}')

  if args.load is not None:
    try:
      enhancement_model.load_state_dict(
        torch.load(args.load, map_location=device, weights_only=True)
      )
      logging.info(f'Model loaded from {args.load}')
    except RuntimeError as e:
      logging.error(
        f'Could not load pretrained weights from {args.load} due to mismatch: {e}'
      )
      sys.exit(1)
    except Exception as e:
      logging.error(f'Unexpected error loading model: {e}')
      sys.exit(1)
  else:
    lr = args.lr

  pipeline_model = PipeLine(running_device=device, photofinishing_model_path=args.photofinishing_model_path)

  denoising_models_postfix = ''
  for denoising_model_path in args.denoising_model_path:

    denoising_models_postfix += os.path.splitext(os.path.split(denoising_model_path)[-1])[0] + '_'

  temp_folder_postfix = denoising_models_postfix + os.path.splitext(
    os.path.split(args.photofinishing_model_path)[-1])[0]

  pipeline_model.to(device=device)
  try:
    train_net(
      model=enhancement_model,
      tr_device=device,
      in_tr_dir=args.in_tr_dir,
      gt_tr_dir=args.gt_tr_dir,
      data_tr_dir=args.data_tr_dir,
      in_val_dir=args.in_vl_dir,
      gt_val_dir=args.gt_vl_dir,
      data_val_dir=args.data_vl_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      lr=args.lr,
      l2_reg=args.l2_r,
      exp_name=model_name,
      validation_frequency=args.val_frq,
      in_sz=args.in_sz,
      temp_folder=args.temp_folder,
      overwrite_temp_folder=args.overwrite_temp_folder,
      delete_temp_folder=args.delete_temp_folder,
      no_tensorboard=args.no_tensorboard,
      denoising_model_paths=args.denoising_model_path,
      temp_folder_postfix=temp_folder_postfix,
      pipeline_model=pipeline_model,
      no_downsampling=args.no_downsampling,
    )
  except KeyboardInterrupt:
    torch.save(enhancement_model.state_dict(), 'interrupted_checkpoint.pth')
    logging.info('Saved interrupt checkpoint backup')
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
