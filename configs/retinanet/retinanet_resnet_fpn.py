_base_ = [
    '../_xbase_/hyper_params/common_config.py',
    '../_xbase_/hyper_params/retinanet_config.py',
    '../_xbase_/hyper_params/schedule_48e.py',
]

dataset_type = 'CocoDataset'

if dataset_type == 'CocoDataset':
    _base_ += ['../_xbase_/datasets/coco_det_1x.py']
    num_classes = 80
elif dataset_type == 'VOCDataset':
    _base_ += ['../_xbase_/datasets/voc0712_det_1x.py']
    num_classes = 20
elif dataset_type == 'CityscapesDataset':
    _base_ += ['../_xbase_/datasets/cityscapes_det_1x.py']
    num_classes = 8
else:
    assert False, f'Unknown dataset_type: {dataset_type}'


input_size = (768,384)          # (1536,768) #(1024,512) #(768,384) #(512,512)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #imagenet mean/std

backbone_type = 'ResNet'
backbone_depth = 50
pretrained='torchvision://resnet50'
bacbone_out_channels=[256, 512, 1024, 2048]
backbone_out_indices = (0, 1, 2, 3)

fpn_type = 'FPN'
fpn_in_channels = bacbone_out_channels
fpn_out_channels = 256
fpn_start_level = 1
fpn_num_outs = 5
fpn_upsample_mode = 'nearest' #'nearest' #'bilinear'
fpn_upsample_cfg = dict(scale_factor=2, mode=fpn_upsample_mode)

#retinanet_base_stride = (8 if fpn_start_level==1 else (4 if fpn_start_level==0 else None))
head_stacked_convs = 4
input_size_divisor = 32

conv_cfg = None
norm_cfg = None

model = dict(
    type='RetinaNet',
    pretrained=pretrained,
    backbone=dict(
        type='ResNet',
        depth=backbone_depth,
        num_stages=4,
        out_indices=backbone_out_indices,
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type=fpn_type,
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        start_level=fpn_start_level,
        num_outs=fpn_num_outs,
        add_extra_convs='on_input',
        upsample_cfg=fpn_upsample_cfg,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=num_classes,
        in_channels=fpn_out_channels,
        stacked_convs=head_stacked_convs,
        feat_channels=fpn_out_channels,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=input_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=input_size_divisor),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=input_size_divisor),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# settings for qat or calibration - uncomment after doing floating point training
# also change dataset_repeats in the dataset config to 1 for fast learning
quantize = False #'training' #'calibration'
if quantize:
  load_from = './data/checkpoints/object_detection/retinanet_resnet_fpn_bgr/latest.pth'
  optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4)
  total_epochs = 1 if quantize == 'calibration' else 5
else:
  optimizer = dict(type='SGD', lr=2e-2, momentum=0.9, weight_decay=1e-4)
#

#load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'