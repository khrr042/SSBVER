import torch
from torch import nn

from . import convnext, mobilenetv3, resnet, resnet_ibn_a, swin_transformer, vit
from .head import ReIDHead, SSLHead
from .wrapper import MultiCropWrapper


def _torch_load(path, map_location=None, weights_only=None):
    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    if weights_only is None:
        return torch.load(path, **kwargs)
    try:
        return torch.load(path, weights_only=weights_only, **kwargs)
    except TypeError:
        # weights_only is unsupported in older PyTorch versions.
        return torch.load(path, **kwargs)


def _build_single_backbone(args, model_arc, is_student=False):
    if 'vit' in model_arc:
        if is_student:
            model = vit.__dict__[model_arc](
                patch_size=args.patch_size, drop_path_rate=args.drop_path_rate)
        else:
            model = vit.__dict__[model_arc](patch_size=args.patch_size)
        embed_dim = model.embed_dim
    elif model_arc == 'resnet50':
        model = resnet.__dict__[model_arc](args)
        embed_dim = model.layer4[2].conv3.weight.shape[0]
    elif model_arc == 'resnet50_ibn_a':
        model = resnet_ibn_a.__dict__[model_arc](args)
        embed_dim = model.layer4[2].conv3.weight.shape[0]
    elif 'swin' in model_arc:
        model = swin_transformer.__dict__[model_arc](args)
        embed_dim = model.norm.weight.shape[0]
    elif 'convnext' in model_arc:
        model = convnext.__dict__[model_arc](pretrained=True)
        embed_dim = model.norm.weight.shape[0]
    elif 'mobilenetv3' in model_arc:
        use_pretrained = is_student and args.pretrained and args.pretrained_method == 'ImageNet'
        model = mobilenetv3.__dict__[model_arc](args=args, pretrained=use_pretrained)
        embed_dim = model.embed_dim
    else:
        raise NameError('Model {} is not applicable!'.format(model_arc))

    return model, embed_dim


def _build_backbones(args):
    teacher_model_arc = args.teacher_model_arc or args.model_arc

    student, student_embed_dim = _build_single_backbone(
        args, args.model_arc, is_student=True)
    teacher_ema, _ = _build_single_backbone(
        args, args.model_arc, is_student=False)
    teacher_frozen, teacher_embed_dim = _build_single_backbone(
        args, teacher_model_arc, is_student=False)
    return student, teacher_ema, teacher_frozen, student_embed_dim, \
        teacher_embed_dim, teacher_model_arc


def _resolve_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in ('teacher_frozen', 'teacher_ema', 'teacher', 'student', 'model', 'state_dict'):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _load_state_dict_flexible(model, state_dict, model_name='model'):
    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []

    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    msg = model.load_state_dict(filtered_state, strict=False)

    if skipped_shape:
        print('{}: skipped {} mismatched tensors while loading checkpoint.'.format(
            model_name, len(skipped_shape)))
        for key, src_shape, dst_shape in skipped_shape[:5]:
            print('  - {}: ckpt {} vs model {}'.format(key, src_shape, dst_shape))
        if len(skipped_shape) > 5:
            print('  - ...')

    if not filtered_state:
        print('{}: WARNING no tensors were loaded. '
              'Checkpoint architecture is likely incompatible.'.format(model_name))

    return msg


def build_models(args, num_classes=None):
    student_backbone, teacher_ema_backbone, teacher_frozen_backbone, \
        student_embed_dim, teacher_embed_dim, teacher_model_arc = \
        _build_backbones(args)

    student_ssl_head = SSLHead(
        in_dim=student_embed_dim,
        out_dim=args.ssl_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer)
    student_reid_head = ReIDHead(
        num_classes=num_classes,
        embed_dim=student_embed_dim,
        neck=args.neck,
        neck_feat=args.neck_feat)

    teacher_ema_ssl_head = SSLHead(
        in_dim=student_embed_dim,
        out_dim=args.ssl_dim,
        use_bn=args.use_bn_in_head)
    teacher_ema_reid_head = ReIDHead(
        num_classes=num_classes,
        embed_dim=student_embed_dim,
        neck=args.neck,
        neck_feat=args.neck_feat)

    teacher_frozen_reid_head = ReIDHead(
        num_classes=num_classes,
        embed_dim=teacher_embed_dim,
        neck=args.neck,
        neck_feat=args.neck_feat)

    student = MultiCropWrapper(
        student_backbone, student_ssl_head, student_reid_head, is_student=True)
    teacher_ema = MultiCropWrapper(
        teacher_ema_backbone,
        teacher_ema_ssl_head,
        teacher_ema_reid_head,
        is_student=False)
    teacher_frozen = MultiCropWrapper(
        teacher_frozen_backbone, None, teacher_frozen_reid_head, is_student=False)

    if args.pretrained and args.pretrained_method == 'ImageNet':
        if args.model_arc == 'vit_small':
            ckpt = _torch_load(
                './models/pretrained_weights/deit_small.pth',
                weights_only=True)['model']
        elif args.model_arc == 'vit_base':
            ckpt = _torch_load(
                './models/pretrained_weights/jx_vit_base_p16_224-80ecf9dd.pth',
                weights_only=True)
        elif args.model_arc == 'resnet50':
            ckpt = _torch_load(
                './models/pretrained_weights/resnet50-0676ba61.pth',
                weights_only=True)
        elif args.model_arc == 'resnet101':
            ckpt = _torch_load(
                './models/pretrained_weights/resnet101-63fe2227.pth',
                weights_only=True)
        elif args.model_arc == 'resnet50_ibn_a':
            ckpt = _torch_load(
                './models/pretrained_weights/resnet50_ibn_a-d9d0bb7b.pth',
                map_location='cpu',
                weights_only=True)
        elif args.model_arc == 'swin_base':
            ckpt = _torch_load(
                './models/pretrained_weights/swin_base_patch4_window7_224_22k.pth',
                map_location='cpu',
                weights_only=True)['model']
        else:
            ckpt = None

        if ckpt is not None:
            m = student.backbone.load_state_dict(ckpt, strict=False)
            print(
                'Student model is loaded by pretrained Weights with this message: {}'
                .format(m))

    student = nn.DataParallel(student.cuda())
    teacher_ema = nn.DataParallel(teacher_ema.cuda())
    teacher_frozen = nn.DataParallel(teacher_frozen.cuda())

    m = teacher_ema.load_state_dict(student.state_dict(), strict=False)
    print(m)

    if args.teacher_frozen_ckpt:
        ckpt = _torch_load(
            args.teacher_frozen_ckpt, map_location='cpu', weights_only=False)
        m = _load_state_dict_flexible(
            teacher_frozen, _resolve_state_dict(ckpt), model_name='teacher_frozen')
        print('Frozen teacher is loaded from {} with message: {}'.format(
            args.teacher_frozen_ckpt, m))
    else:
        if teacher_model_arc == args.model_arc:
            m = teacher_frozen.load_state_dict(student.state_dict(), strict=False)
            print('Frozen teacher initialized from student with message: {}'.format(m))
        else:
            print('Frozen teacher initialized randomly because teacher_model_arc ({}) '
                  'differs from model_arc ({}) and no teacher_frozen_ckpt was provided.'
                  .format(teacher_model_arc, args.model_arc))

    for p in teacher_ema.parameters():
        p.requires_grad = False

    for p in teacher_frozen.parameters():
        p.requires_grad = (args.train_mode == 'teacher_exp_only')

    print('student, teacher_ema and teacher_frozen models are ready!')
    return student, teacher_ema, teacher_frozen
