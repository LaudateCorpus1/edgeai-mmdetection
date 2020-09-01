#!/usr/bin/env python

import sys
import os
import datetime

########################################################################
# config
from detection_configs import config

########################################################################
base_path = os.path.splitext(os.path.basename(config))[0]
work_dir = os.path.join('./work_dirs', base_path)
checkpoint = f'{work_dir}/latest.pth'
outfile = os.path.join(work_dir,'model.onnx')
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
shape = [512, 512]
opset_version = 9

print(f'Exporting ONNX with: {config} @ {date}')

from xmmdet.tools import pytorch2onnx
sys.argv = [sys.argv[0], f'{config}', f'{checkpoint}', f'--out={outfile}', f'--shape', f'{shape[0]}', f'{shape[1]}', f'--opset_version', f'{opset_version}']
args = pytorch2onnx.parse_args()
pytorch2onnx.main(args)

