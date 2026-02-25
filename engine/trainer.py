from time import time
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter

from .utils import train_one_epoch
from .evaluator import do_eval

def train(args,
            train_loader,
            val_loader,
            student,
            teacher_ema,
            teacher_frozen,
            loss_fn,
            optimizer,
            lr_scheduler,
            logger,
            start_epoch=0):
    
    logger.info('Initial Evaluation ...')
    writer = SummaryWriter(osp.join(args.output_dir, 'TensorBoard'))
    best_map, best_cmc1, best_cmc5 = 0, 0, 0
    best_map_epoch, best_cmc1_epoch, best_cmc5_epoch = 0, 0, 0
    
    best_map_ema, best_cmc1_ema, best_cmc5_ema = 0, 0, 0
    best_map_epoch_ema, best_cmc1_epoch_ema, best_cmc5_epoch_ema = 0, 0, 0
    
    if args.train_mode == 'teacher_exp_only':
        mAP, CMC1, CMC5 = do_eval(args, val_loader, teacher_frozen, logger)
        writer.add_scalar('Accuracy-teacher_exp/mAP', mAP, start_epoch)
        writer.add_scalar('Accuracy-teacher_exp/CMC1', CMC1, start_epoch)
        writer.add_scalar('Accuracy-teacher_exp/CMC5', CMC5, start_epoch)
    else:
        mAP, CMC1, CMC5 = do_eval(args, val_loader, teacher_ema, logger, EMA=True)
        writer.add_scalar('Accuracy-teacher/mAP', mAP, start_epoch)
        writer.add_scalar('Accuracy-teacher/CMC1', CMC1, start_epoch)
        writer.add_scalar('Accuracy-teacher/CMC5', CMC5, start_epoch)
    
    logger.info('Start Training ...')
    start = time()
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(args,
            epoch,
            train_loader,
            student,
            teacher_ema,
            teacher_frozen,
            loss_fn,
            optimizer,
            lr_scheduler,
            logger,
            writer)
        
        if (epoch + 1) % args.eval_freq == 0:
            if args.train_mode == 'teacher_exp_only':
                mAP_exp, CMC1_exp, CMC5_exp = do_eval(
                                args, val_loader, teacher_frozen, logger)
                writer.add_scalar('Accuracy-teacher_exp/mAP', mAP_exp, epoch + 1)
                writer.add_scalar('Accuracy-teacher_exp/CMC1', CMC1_exp, epoch + 1)
                writer.add_scalar('Accuracy-teacher_exp/CMC5', CMC5_exp, epoch + 1)

                if mAP_exp > best_map_ema:
                    best_map_ema, best_map_epoch_ema = mAP_exp, epoch + 1

                if CMC1_exp > best_cmc1_ema:
                    best_cmc1_ema, best_cmc1_epoch_ema = CMC1_exp, epoch + 1

                if CMC5_exp > best_cmc5_ema:
                    best_cmc5_ema, best_cmc5_epoch_ema = CMC5_exp, epoch + 1
            else:
                mAP_ema, CMC1_ema, CMC5_ema = do_eval(
                                args, val_loader, teacher_ema, logger, EMA=True)
                writer.add_scalar('Accuracy-teacher/mAP', mAP_ema, epoch + 1)
                writer.add_scalar('Accuracy-teacher/CMC1', CMC1_ema, epoch + 1)
                writer.add_scalar('Accuracy-teacher/CMC5', CMC5_ema, epoch + 1)

                mAP, CMC1, CMC5 = do_eval(args, val_loader, student, logger)
                writer.add_scalar('Accuracy-student/mAP', mAP, epoch + 1)
                writer.add_scalar('Accuracy-student/CMC1', CMC1, epoch + 1)
                writer.add_scalar('Accuracy-student/CMC5', CMC5, epoch + 1)

                if mAP > best_map:
                    best_map, best_map_epoch = mAP, epoch + 1

                if CMC1 > best_cmc1:
                    best_cmc1, best_cmc1_epoch = CMC1, epoch + 1

                if CMC5 > best_cmc5:
                    best_cmc5, best_cmc5_epoch = CMC5, epoch + 1

                if mAP_ema > best_map_ema:
                    best_map_ema, best_map_epoch_ema = mAP_ema, epoch + 1

                if CMC1_ema > best_cmc1_ema:
                    best_cmc1_ema, best_cmc1_epoch_ema = CMC1_ema, epoch + 1

                if CMC5_ema > best_cmc5_ema:
                    best_cmc5_ema, best_cmc5_epoch_ema = CMC5_ema, epoch + 1

        if (epoch + 1) % args.save_ckpt_freq == 0:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher_ema.state_dict(),
                'teacher_ema': teacher_ema.state_dict(),
                'teacher_frozen': teacher_frozen.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args}
            torch.save(save_dict, 
                    osp.join(args.output_dir, 
                            'ckpt_epoch_{0:03d}.pth'.format(epoch + 1)))
            
    end = time()
    hours = int((end - start)) // 3600
    minutes = int((float((end - start) / 3600.) - hours) * 60)
    logger.info('Finished Training in {0:d} Hours and {1:d} Minutes'.format(
        hours, minutes))
    
    logger.info('------ Student model Results ------:')
    logger.info('Best mAP: {0:.2%} @ epoch : {1:}'.format(
                                                best_map, best_map_epoch))
    logger.info('Best CMC1: {0:.2%} @ epoch : {1:}'.format(
                                                best_cmc1, best_cmc1_epoch))
    logger.info('Best CMC5: {0:.2%} @ epoch : {1:}'.format(
                                                best_cmc5, best_cmc5_epoch))
    logger.info('------ Teacher model Results ------:')
    logger.info('Best mAP_EMA: {0:.2%} @ epoch : {1:}'.format(
                                            best_map_ema, best_map_epoch_ema))
    logger.info('Best CMC1_EMA: {0:.2%} @ epoch : {1:}'.format(
                                            best_cmc1_ema, best_cmc1_epoch_ema))
    logger.info('Best CMC5_EMA: {0:.2%} @ epoch : {1:}'.format(
                                            best_cmc5_ema, best_cmc5_epoch_ema))




        
        
