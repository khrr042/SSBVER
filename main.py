import argparse
import os
import warnings

from configs import get_configs
from tools import log


def _torch_load(path, map_location=None, weights_only=None):
    import torch

    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    if weights_only is None:
        return torch.load(path, **kwargs)
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    from tools import utils
    from data import build_data
    from models import (
        build_models,
        load_checkpoint_for_eval,
        resume_training_from_checkpoint,
    )
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
    student, teacher_ema, teacher_frozen = build_models(
        args, num_classes=num_train_classes)

    if args.is_train:
        loss_fn = build_loss_fn(args, num_classes=num_train_classes)
        target_model = teacher_frozen if args.train_mode == 'teacher_exp_only' \
            else student
        optimizer, lr_scheduler = build_solver(args, target_model)
        start_epoch = resume_training_from_checkpoint(
            args,
            student,
            teacher_ema,
            teacher_frozen,
            optimizer=optimizer,
            logger=logger)

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
        from engine.evaluator import do_eval
        ckpt = _torch_load(args.test_ckpt, map_location='cpu', weights_only=False)
        model_key = args.test_model
        if model_key == 'teacher' and 'teacher_ema' in ckpt:
            model_key = 'teacher_ema'
        if model_key == 'teacher_ema' and 'teacher_ema' not in ckpt and 'teacher' in ckpt:
            model_key = 'teacher'
        if model_key not in ckpt:
            raise KeyError('Model key {} is not found in checkpoint {}'.format(
                model_key, args.test_ckpt))
        load_checkpoint_for_eval(
            student,
            ckpt[model_key],
            model_name=model_key,
            logger=logger)
        print('{} model is loaded from {}'.format(args.test_model, args.test_ckpt))
        do_eval(args=args,
                val_loader=val_loader,
                model=student)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='SSLBVER', 
                                        parents=[get_configs()])
    main(parser.parse_args())



