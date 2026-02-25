import torch
from torch import nn

from . import convnext, mobilenetv3, resnet, resnet_ibn_a, swin_transformer, vit
from .head import ReIDHead, SSLHead
from .wrapper import MultiCropWrapper


def _build_backbone(arc, args, is_student, use_constructor_pretrained=False):
    if "vit" in arc:
        kwargs = {"patch_size": args.patch_size}
        if is_student:
            kwargs["drop_path_rate"] = args.drop_path_rate
        backbone = vit.__dict__[arc](**kwargs)
        embed_dim = backbone.embed_dim
        return backbone, embed_dim

    if arc in ("resnet50", "resnet101"):
        backbone = resnet.__dict__[arc](args)
        embed_dim = backbone.layer4[2].conv3.weight.shape[0]
        return backbone, embed_dim

    if arc == "resnet50_ibn_a":
        backbone = resnet_ibn_a.__dict__[arc](args)
        embed_dim = backbone.layer4[2].conv3.weight.shape[0]
        return backbone, embed_dim

    if "swin" in arc:
        backbone = swin_transformer.__dict__[arc](args)
        embed_dim = backbone.norm.weight.shape[0]
        return backbone, embed_dim

    if "convnext" in arc:
        backbone = convnext.__dict__[arc](pretrained=use_constructor_pretrained)
        embed_dim = backbone.norm.weight.shape[0]
        return backbone, embed_dim

    if "mobilenetv3" in arc:
        backbone = mobilenetv3.__dict__[arc](args=args, pretrained=use_constructor_pretrained)
        embed_dim = backbone.embed_dim
        return backbone, embed_dim
    
    

    raise ValueError("Unsupported model architecture: {}".format(arc))


def _load_local_imagenet_weights(backbone, arc):
    if arc == "vit_small":
        ckpt = torch.load("./models/pretrained_weights/deit_small.pth", map_location="cpu")["model"]
    elif arc == "vit_base":
        ckpt = torch.load("./models/pretrained_weights/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
    elif arc == "resnet50":
        ckpt = torch.load("./models/pretrained_weights/resnet50-0676ba61.pth", map_location="cpu")
    elif arc == "resnet101":
        ckpt = torch.load("./models/pretrained_weights/resnet101-63fe2227.pth", map_location="cpu")
    elif arc == "resnet50_ibn_a":
        ckpt = torch.load("./models/pretrained_weights/resnet50_ibn_a-d9d0bb7b.pth", map_location="cpu")
    elif arc == "swin_base":
        ckpt = torch.load(
            "./models/pretrained_weights/swin_base_patch4_window7_224_22k.pth",
            map_location="cpu",
        )["model"]
    else:
        return False, "No local ImageNet checkpoint mapping for {}".format(arc)

    msg = backbone.load_state_dict(ckpt, strict=False)
    return True, msg


def _extract_state_dict_from_ckpt(ckpt):
    candidate_keys = ["teacher_frozen", "teacher_exp", "teacher", "teacher_ema", "student", "model"]
    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    return None


def build_models(args, num_classes=None):
    student_arc = args.student_arc if args.student_arc else args.model_arc
    teacher_exp_arc = args.teacher_exp_arc if args.teacher_exp_arc else student_arc

    constructor_pretrained_arcs = ("convnext", "mobilenetv3")

    student_backbone, student_embed_dim = _build_backbone(
        student_arc,
        args,
        is_student=True,
        use_constructor_pretrained=args.pretrained and any(x in student_arc for x in constructor_pretrained_arcs),
    )
    teacher_ema_backbone, _ = _build_backbone(
        student_arc,
        args,
        is_student=False,
        use_constructor_pretrained=False,
    )
    teacher_exp_backbone, teacher_exp_embed_dim = _build_backbone(
        teacher_exp_arc,
        args,
        is_student=False,
        use_constructor_pretrained=args.teacher_exp_pretrained and any(
            x in teacher_exp_arc for x in constructor_pretrained_arcs
        ),
    )

    student_ssl_head = SSLHead(
        in_dim=student_embed_dim,
        out_dim=args.ssl_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )
    student_reid_head = ReIDHead(
        num_classes=num_classes,
        embed_dim=student_embed_dim,
        neck=args.neck,
        neck_feat=args.neck_feat,
    )

    teacher_ssl_head = SSLHead(
        in_dim=student_embed_dim,
        out_dim=args.ssl_dim,
        use_bn=args.use_bn_in_head,
    )
    teacher_reid_head_ema = ReIDHead(
        num_classes=num_classes,
        embed_dim=student_embed_dim,
        neck=args.neck,
        neck_feat=args.neck_feat,
    )
    teacher_reid_head_frozen = ReIDHead(
        num_classes=num_classes,
        embed_dim=teacher_exp_embed_dim,
        neck=args.neck,
        neck_feat=args.neck_feat,
    )

    if student_embed_dim != teacher_exp_embed_dim:
        kd_projector = nn.Linear(student_embed_dim, teacher_exp_embed_dim, bias=False)
        print(
            "Enable KD projector for dim mismatch: student {} -> teacher_exp {}".format(
                student_embed_dim, teacher_exp_embed_dim
            )
        )
    else:
        kd_projector = nn.Identity()

    student = MultiCropWrapper(
        student_backbone,
        student_ssl_head,
        student_reid_head,
        kd_projector=kd_projector,
    )
    teacher_ema = MultiCropWrapper(
        teacher_ema_backbone,
        teacher_ssl_head,
        teacher_reid_head_ema,
        is_student=False,
    )
    teacher_frozen = MultiCropWrapper(
        teacher_exp_backbone,
        None,
        teacher_reid_head_frozen,
        is_student=False,
    )

    if (
        args.pretrained
        and args.pretrained_method == "ImageNet"
        and not any(x in student_arc for x in constructor_pretrained_arcs)
    ):
        loaded, msg = _load_local_imagenet_weights(student.backbone, student_arc)
        if loaded:
            print("Student model is loaded by ImageNet weights with message: {}".format(msg))
        else:
            print(msg)

    if (
        args.teacher_exp_pretrained
        and args.pretrained_method == "ImageNet"
        and not args.frozen_teacher_ckpt
        and not any(x in teacher_exp_arc for x in constructor_pretrained_arcs)
    ):
        loaded, msg = _load_local_imagenet_weights(teacher_frozen.backbone, teacher_exp_arc)
        if loaded:
            print("Frozen expert is loaded by ImageNet weights with message: {}".format(msg))
        else:
            print(msg)

    student = student.cuda()
    teacher_ema = teacher_ema.cuda()
    teacher_frozen = teacher_frozen.cuda()

    student = nn.DataParallel(student)
    teacher_ema = nn.DataParallel(teacher_ema)
    teacher_frozen = nn.DataParallel(teacher_frozen)

    m = teacher_ema.load_state_dict(student.state_dict(), strict=False)
    print("EMA teacher init from student: {}".format(m))

    if teacher_exp_arc == student_arc and not args.frozen_teacher_ckpt:
        m = teacher_frozen.load_state_dict(student.state_dict(), strict=False)
        print("Frozen teacher init from student (same arc): {}".format(m))

    if args.frozen_teacher_ckpt:
        ckpt = torch.load(args.frozen_teacher_ckpt, map_location="cpu")
        state_dict = _extract_state_dict_from_ckpt(ckpt)
        if state_dict is None:
            raise ValueError("Cannot parse checkpoint state_dict from {}".format(args.frozen_teacher_ckpt))
        m = teacher_frozen.load_state_dict(state_dict, strict=False)
        print("Frozen teacher is loaded from {} with message: {}".format(args.frozen_teacher_ckpt, m))

    for p in teacher_ema.parameters():
        p.requires_grad = False
    for p in teacher_frozen.parameters():
        p.requires_grad = args.train_mode == "teacher_exp_only"

    print(
        "student({}), teacher_ema({}) and teacher_frozen({}) are ready!".format(
            student_arc, student_arc, teacher_exp_arc
        )
    )
    return student, teacher_ema, teacher_frozen
