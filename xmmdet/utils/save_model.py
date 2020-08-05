import os
import torch
from .quantize import is_mmdet_quant_module
from .proto import mmdet_meta_arch_pb2
from google.protobuf import text_format

__all__ = ['save_model_proto']


def save_model_proto(cfg, model, input, output_filename):
    is_cuda = next(model.parameters()).is_cuda
    input_list = input if isinstance(input, torch.Tensor) else _create_rand_inputs(input, is_cuda)
    input_size = input.size() if isinstance(input, torch.Tensor) else input
    model = model.module if is_mmdet_quant_module(model) else model
    is_ssd = hasattr(cfg.model, 'bbox_head') and ('SSD' in cfg.model.bbox_head.type)
    is_fcos = hasattr(cfg.model, 'bbox_head') and ('FCOS' in cfg.model.bbox_head.type)
    is_retinanet = hasattr(cfg.model, 'bbox_head') and ('Retina' in cfg.model.bbox_head.type)
    if is_ssd:
        input_names = ['input']
        output_names = []
        for cls_idx, cls in enumerate(model.bbox_head.cls_convs):
            output_names.append(f'cls_convs_{cls_idx}')
        #
        for reg_idx, reg in enumerate(model.bbox_head.reg_convs):
            output_names.append(f'reg_convs_{reg_idx}')
        #
        _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names, output_names)
        _save_mmdet_proto_ssd(cfg, model, input_size, output_filename, input_names, output_names)
    elif is_retinanet:
        input_names = ['input']
        output_names = []
        for i in range(model.neck.num_outs):
            output_names.append(f'retina_cls_{i}')
        #
        for i in range(model.neck.num_outs):
            output_names.append(f'retina_reg_{i}')
        #
        _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names, output_names)
        _save_mmdet_proto_retinanet(cfg, model, input_size, output_filename, input_names, output_names)
    elif is_fcos:
        input_names = ['input']
        output_names = []
        for i in range(model.neck.num_outs):
            output_names.append(f'cls_convs_{i}')
        #
        for i in range(model.neck.num_outs):
            output_names.append(f'reg_convs_{i}')
        #
        for i in range(model.neck.num_outs):
            output_names.append(f'centerness_convs_{i}')
        #
        _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names, output_names)
    else:
        _save_mmdet_onnx(cfg, model, input_list, output_filename)
    #


###########################################################
def _create_rand_inputs(input_size, is_cuda=False):
    x = torch.rand(input_size)
    x = x.cuda() if is_cuda else x
    return x


def _save_mmdet_onnx(cfg, model, input_list, output_filename, input_names=None, output_names=None):
    #https://github.com/open-mmlab/mmdetection/pull/1082
    assert hasattr(model, 'forward_dummy'), 'wrting onnx is not supported by this model'
    model.eval()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    forward_backup = model.forward #backup forward
    model.forward = model.forward_dummy #set dummy forward
    opset_version = 9
    torch.onnx.export(model, input_list, output_filename, input_names=input_names,
                      output_names=output_names, export_params=True, verbose=False, opset_version=opset_version)
    model.forward = forward_backup #restore forward


###########################################################
def _save_mmdet_proto_ssd(cfg, model, input_size, output_filename, input_names=None, output_names=None):
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    num_output_names = len(output_names)//2
    cls_output_names = output_names[:num_output_names]
    reg_output_names = output_names[num_output_names:]
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator

    prior_box_param = []
    for h_idx in range(num_output_names):
        min_size=[anchor_generator.min_sizes[h_idx]]
        max_size=[anchor_generator.max_sizes[h_idx]]
        aspect_ratio=anchor_generator.ratios[h_idx][2::2]
        step=anchor_generator.strides[h_idx]
        step=step[0] if isinstance(step,(tuple,list)) else step
        prior_box_param.append(mmdet_meta_arch_pb2.PriorBoxParameter(min_size=min_size, max_size=max_size,
                                                                     aspect_ratio=aspect_ratio, step=step,
                                                                     variance=bbox_head.bbox_coder.stds, clip=False, flip=True))
    #

    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=100)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=bbox_head.num_classes+1, share_location=True,
                                            background_label_id=bbox_head.num_classes, nms_param=nms_param,
                                            code_type=mmdet_meta_arch_pb2.CENTER_SIZE, keep_top_k=100,
                                            confidence_threshold=0.5)

    ssd = mmdet_meta_arch_pb2.TidlMaCaffeSsd(box_input=reg_output_names, class_input=cls_output_names, output='output', prior_box_param=prior_box_param,
                                             in_width=input_size[3], in_height=input_size[2], detection_output_param=detection_output_param)

    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='ssd',  caffe_ssd=[ssd])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)


###########################################################
def _save_mmdet_proto_retinanet(cfg, model, input_size, output_filename, input_names=None, output_names=None, name='model.prototxt'):
    output_filename = os.path.splitext(output_filename)[0] + '.prototxt'
    num_output_names = len(output_names)//2
    cls_output_names = output_names[:num_output_names]
    reg_output_names = output_names[num_output_names:]
    bbox_head = model.bbox_head
    anchor_generator = bbox_head.anchor_generator

    background_label_id = -1 if bbox_head.use_sigmoid_cls else bbox_head.num_classes
    num_classes = bbox_head.num_classes if bbox_head.use_sigmoid_cls else bbox_head.num_classes+1
    score_converter = mmdet_meta_arch_pb2.SIGMOID if bbox_head.use_sigmoid_cls else mmdet_meta_arch_pb2.SOFTMAX
    anchor_param = mmdet_meta_arch_pb2.RetinaNetAnchorParameter(aspect_ratio=anchor_generator.ratios,
                                                                octave_base_scale=anchor_generator.octave_base_scale,
                                                                scales_per_octave=anchor_generator.scales_per_octave)

    nms_param = mmdet_meta_arch_pb2.TIDLNmsParam(nms_threshold=0.45, top_k=100)
    detection_output_param = mmdet_meta_arch_pb2.TIDLOdPostProc(num_classes=num_classes, share_location=True,
                                            background_label_id=background_label_id, nms_param=nms_param,
                                            code_type=mmdet_meta_arch_pb2.CENTER_SIZE, keep_top_k=100,
                                            confidence_threshold=0.5)

    retinanet = mmdet_meta_arch_pb2.TidlMaRetinaNet(box_input=reg_output_names, class_input=cls_output_names, output='output',
                                              x_scale=1.0, y_scale=1.0, width_scale=1.0, height_scale=1.0,
                                              in_width=input_size[3], in_height=input_size[2],
                                              score_converter=score_converter, anchor_param=anchor_param,
                                              detection_output_param=detection_output_param)

    arch = mmdet_meta_arch_pb2.TIDLMetaArch(name='retinanet',  tidl_retinanet=[retinanet])

    with open(output_filename, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)