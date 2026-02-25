import argparse
import os
import warnings

from configs import get_configs
from tools import log

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    from tools import utils
    from data import build_data
    from models import build_models
    from layers import build_loss_fn
    from solver import build_solver
    from engine.trainer import train

    utils.fix_random_seeds(seed=args.seed)
    utils.create_folder(args)
    utils.save_configs(args)

    logger = log.setup_logger(args)
    if args.is_train:
        logger.info("Using {} GPUs".format(len(args.device_ids.split(','))))
        logger.info('Training Configurations:')
        for k, v in args.__dict__.items():
            logger.info('{}: {}'.format(k, v))
        
    train_loader, val_loader, num_train_classes = build_data(args)
    student, teacher_ema, teacher_frozen = build_models(args, num_classes=num_train_classes)

    if args.is_train:
        import torch
        if args.train_mode == 'teacher_exp_only':
            args.ssl_loss_lambda = 0.0
            args.cmpt_loss_lambda = 0.0
            args.kd_loss_lambda = 0.0
            logger.info('teacher_exp_only mode: disable SSL/CMPT/KD losses.')
        loss_fn = build_loss_fn(args, num_classes=num_train_classes)
        optim_model = teacher_frozen if args.train_mode == 'teacher_exp_only' else student
        optimizer, lr_scheduler = build_solver(args, optim_model)
        start_epoch = 0

        if args.resume_ckpt:
            ckpt = torch.load(args.resume_ckpt, map_location='cpu', weights_only=False)
            if 'student' in ckpt:
                m = student.load_state_dict(ckpt['student'], strict=False)
                logger.info('Student is resumed from {} with message: {}'.format(
                    args.resume_ckpt, m))

            teacher_ema_key = 'teacher_ema' if 'teacher_ema' in ckpt else 'teacher'
            if teacher_ema_key in ckpt:
                m = teacher_ema.load_state_dict(ckpt[teacher_ema_key], strict=False)
                logger.info('Teacher EMA is resumed from {} with message: {}'.format(
                    args.resume_ckpt, m))

            if 'teacher_frozen' in ckpt:
                m = teacher_frozen.load_state_dict(ckpt['teacher_frozen'], strict=False)
                logger.info('Frozen teacher is resumed from {} with message: {}'.format(
                    args.resume_ckpt, m))

            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                logger.info('Optimizer is resumed from {}'.format(args.resume_ckpt))

            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
                logger.info('Resume start epoch: {} (checkpoint epoch: {})'.format(
                    start_epoch, ckpt['epoch']))
            if start_epoch >= args.epochs:
                logger.info('Checkpoint already reached target epochs (start_epoch={}, epochs={}). Nothing to train.'.format(
                    start_epoch, args.epochs))
                return

        train(args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                student=student,
                teacher_ema=teacher_ema,
                teacher_frozen=teacher_frozen,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                logger=logger,
                start_epoch=start_epoch)
    
    else:
        import torch
        from engine.evaluator import do_eval
        ckpt = torch.load(args.test_ckpt, weights_only=False)
        test_model_key = args.test_model
        if args.test_model == 'teacher' and 'teacher' not in ckpt and 'teacher_ema' in ckpt:
            test_model_key = 'teacher_ema'
        student.load_state_dict(ckpt[test_model_key], strict=False)
        print('{} model is loaded from {}'.format(args.test_model, args.test_ckpt))
        do_eval(args=args,
                val_loader=val_loader,
                model=student)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='SSLBVER', 
                                        parents=[get_configs()])
    main(parser.parse_args())



