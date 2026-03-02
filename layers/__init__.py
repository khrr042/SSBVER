import torch.nn.functional as F
from torch import nn

from .loss import (
    CMPTLoss,
    CrossEntropyLabelSmooth,
    FeatureKDLoss,
    LogitKDLoss,
    SSLLoss,
    TripletLoss,
)


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
        kd_feat_loss = FeatureKDLoss(loss_type=args.kd_loss_type)
        kd_logit_loss = LogitKDLoss(temperature=args.kd_temp)
    else:
        kd_feat_loss = None
        kd_logit_loss = None

    def loss_fn(
        cls_score,
        feat,
        target,
        student_out,
        teacher_out,
        student_kd_feat,
        teacher_kd_feat,
        teacher_kd_logits,
        epoch,
    ):
        id_loss = args.id_loss_lambda * cls_loss(cls_score, target)
        trip_loss = args.triplet_loss_lambda * triplet_loss(feat, target)
        
        if args.ssl_loss_lambda > 0:
            ssl_loss_ssl = args.ssl_loss_lambda * ssl_loss(student_out, teacher_out, epoch)
        else:
            ssl_loss_ssl = None
        
        if args.cmpt_loss_lambda > 0:
            ssl_loss_cmpt = args.cmpt_loss_lambda * cmpt_loss(student_out, teacher_out)
        else:
            ssl_loss_cmpt = None
        
        kd_loss_value = None
        if args.kd_loss_lambda > 0 and teacher_kd_feat is not None:
            kd_feat_value = kd_feat_loss(student_kd_feat, teacher_kd_feat)

            # Optional logit KD: use only when enabled and teacher logits are available.
            if args.kd_alpha > 0 and teacher_kd_logits is not None:
                kd_logit_value = kd_logit_loss(cls_score, teacher_kd_logits)
                kd_mix = args.kd_alpha * kd_logit_value + (1 - args.kd_alpha) * kd_feat_value
            else:
                kd_mix = kd_feat_value

            kd_loss_value = args.kd_loss_lambda * kd_mix
        
        return id_loss, trip_loss, ssl_loss_ssl, ssl_loss_cmpt, kd_loss_value, ssl_loss
    
    return loss_fn
