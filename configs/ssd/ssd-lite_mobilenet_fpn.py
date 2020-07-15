_base_ = [
    '../_xbase_/hyper_params/common_config.py',
    '../_xbase_/hyper_params/ssd_config.py',
    '../_xbase_/hyper_params/schedule_60e.py',
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


input_size = (768,384)          #(1536,768) #(1024,512) #(768,384) #(512,512)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #imagenet mean/std

backbone_type = 'MobileNetV2' #'MobileNetV2' #'MobileNetV1'
mobilenetv2_pretrained='torchvision://mobilenet_v2'
mobilenetv1_pretrained='./data/modelzoo/pytorch/image_classification/imagenet1k/jacinto_ai/mobilenet_v1_2019-09-06_17-15-44.pth'
pretrained=(mobilenetv2_pretrained if backbone_type == 'MobileNetV2' else mobilenetv1_pretrained)
bacbone_out_channels=[24,32,96,320] if backbone_type == 'MobileNetV2' else [128,256,512,1024]
backbone_out_indices = (1, 2, 3, 4)

fpn_in_channels = bacbone_out_channels
fpn_out_channels = 256
fpn_start_level = 1
fpn_num_outs = 6
fpn_upsample_mode = 'bilinear' #'nearest' #'bilinear'
fpn_upsample_cfg = dict(scale_factor=2, mode=fpn_upsample_mode)

basesize_ratio_range = (0.1, 0.9)

conv_cfg = dict(type='ConvDWSep')
norm_cfg = dict(type='BN')

model = dict(
    type='SingleStageDetector',
    pretrained=pretrained,
    backbone=dict(
        type=backbone_type,
        strides=(2, 2, 2, 2, 2),
        depth=None,
        with_last_pool=False,
        ceil_mode=True,
        need_extra=False,
        out_indices=backbone_out_indices,
        out_feature_indices=None,
        l2_norm_scale=None),
    neck=dict(
        type='FPNLite',
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        start_level=fpn_start_level,
        num_outs=fpn_num_outs,
        add_extra_convs='on_input',
        upsample_cfg=fpn_upsample_cfg,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='SSDLiteHead',
        in_channels=[fpn_out_channels for _ in range(6)],
        num_classes=num_classes,
        conv_cfg=conv_cfg,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=basesize_ratio_range,
            strides=[8, 16, 32, 64, 128, 256],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=input_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
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
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# settings for qat or calibration - uncomment after doing floating point training
# also change dataset_repeats in the dataset config to 1 for fast learning
quantize = False #'training' #'calibration'
if quantize:
  load_from = './data/checkpoints/object_detection/ssd-lite_mobilenet_fpn/latest.pth'
  optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4)
  total_epochs = 1 if quantize == 'calibration' else 5
else:
  optimizer = dict(type='SGD', lr=4e-2, momentum=0.9, weight_decay=4e-5) #1e-4 => 4e-5
#
