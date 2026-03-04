import torch.nn.functional as F
from torch import nn

from .loss import (CMPTLoss, SSLLoss, CrossEntropyLabelSmooth, FeatureKDLoss,
                   PairwiseKDLoss, TripletLoss)


def build_loss_fn(args, num_classes=None):
    if args.label_smoothing:
        cls_loss = CrossEntropyLabelSmooth(num_classes=num_classes,
                                            epsilon=args.label_smoothing_eps)
    
    else:
        cls_loss = nn.CrossEntropyLoss()

    triplet_loss = TripletLoss(use_margin=args.use_margin,
                                margin=args.triplet_loss_margin)
    
    if args.ssl_loss_lambda > 0:
        ssl_loss = SSLLoss(
                    out_dim=args.ssl_dim,
                    ncrops=args.local_crops_num + 2,
                    warmup_teacher_temp=args.warmup_teacher_temp,
                    teacher_temp=args.teacher_temp,
                    warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
                    student_temp=args.student_temp,
                    nepochs=args.epochs)
    else:
        ssl_loss = None
    
    if args.cmpt_loss_lambda:
        cmpt_loss = CMPTLoss(ncrops=args.local_crops_num + 2)

    if args.kd_loss_lambda > 0:
        feature_kd_loss = FeatureKDLoss(loss_type=args.kd_loss_type)
        pairwise_kd_loss = PairwiseKDLoss(loss_type=args.kd_pairwise_type)
    else:
        feature_kd_loss = None
        pairwise_kd_loss = None

    def _kd_warmup_scale(epoch):
        if args.kd_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(args.kd_warmup_epochs))

    def loss_fn(cls_score,
                feat,
                target,
                student_out,
                teacher_out,
                epoch,
                student_kd_feat=None,
                teacher_kd_feat=None):
        id_loss = args.id_loss_lambda * cls_loss(cls_score, target)
        trip_loss = args.triplet_loss_lambda * triplet_loss(feat, target)
        
        if args.ssl_loss_lambda > 0 and student_out is not None and teacher_out is not None:
            ssl_loss_ssl = args.ssl_loss_lambda * ssl_loss(student_out, teacher_out, epoch)
        else:
            ssl_loss_ssl = None
        
        if args.cmpt_loss_lambda > 0 and student_out is not None and teacher_out is not None:
            ssl_loss_cmpt = args.cmpt_loss_lambda * cmpt_loss(student_out, teacher_out)
        else:
            ssl_loss_cmpt = None

        kd_feat = cls_score.new_tensor(0.0)
        kd_pairwise = cls_score.new_tensor(0.0)
        kd_total = cls_score.new_tensor(0.0)
        if args.kd_loss_lambda > 0 and student_kd_feat is not None and teacher_kd_feat is not None:
            if args.kd_feature_loss_lambda > 0:
                kd_feat = feature_kd_loss(student_kd_feat, teacher_kd_feat)
            if args.kd_pairwise_loss_lambda > 0:
                kd_pairwise = pairwise_kd_loss(student_kd_feat, teacher_kd_feat)
            kd_inner = args.kd_feature_loss_lambda * kd_feat + \
                args.kd_pairwise_loss_lambda * kd_pairwise
            kd_total = args.kd_loss_lambda * _kd_warmup_scale(epoch) * kd_inner

        return id_loss, trip_loss, ssl_loss_ssl, ssl_loss_cmpt, kd_total, \
            kd_feat, kd_pairwise, ssl_loss
    
    return loss_fn
