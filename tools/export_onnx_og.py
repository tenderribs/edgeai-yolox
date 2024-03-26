#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn
import onnx
import json

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module, PostprocessExport
from yolox.data.data_augment import preproc as preprocess


import cv2
_SUPPORTED_DATASETS = ["coco", "lm","lmo", "ycbv", "coco_kpts", "tree_kpts"]
_NUM_CLASSES = {"coco":80, "lm":15, "lmo":8, "ycbv": 21, "coco_kpts":1, "tree_kpts": 1}
_VAL_ANN = {
    "coco":"instances_val2017.json",
    "lm":"instances_test.json",
    "lmo":"instances_test_bop.json",
    "ycbv": "instances_test_bop.json",
    "coco_kpts": "person_keypoints_val2017.json",
    "tree_kpts": "trees_val.json",
}
_TRAIN_ANN = {
    "coco":"instances_train2017.json",
    "lm":"instances_train.json",
    "lmo":"instances_train.json",
    "ycbv": "instances_train.json",
    "coco_kpts": "person_keypoints_train2017.json",
    "tree_kpts": "trees_train.json",
}
_SUPPORTED_TASKS = {
    "coco":["2dod"],
    "lm":["2dod", "object_pose"],
    "lmo":["2dod", "object_pose"],
    "ycbv":["2dod", "object_pose"],
    "coco_kpts": ["2dod", "human_pose"],
    "tree_kpts": ["2dod", "human_pose"],
}

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--dataset", default=None, type=str, help="dataset for training")
    parser.add_argument("--task", default=None, type=str, help="type of task for model eval")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("--export-det",  action='store_true', help='export the nms part in ONNX model')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--train_ann",
        help="train annotation file name",
        default=None,
    )
    parser.add_argument(
        "--val_ann",
        help="val annotation file name",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser




@logger.catch
def main(kwargs=None, exp=None):
    args = make_parser().parse_args()
    if kwargs is not None:
        for k, v in kwargs.items():
            setattr(args, k, v)
    logger.info("args value: {}".format(args))
    exp = exp if exp is not None else get_exp(args.exp_file, args.name)
    if args.dataset is not None:
        assert (
            args.dataset in _SUPPORTED_DATASETS
        ), "The given dataset is not supported for training!"
        exp.data_set = args.dataset
        exp.num_classes = _NUM_CLASSES[args.dataset]
        exp.val_ann = args.val_ann or _VAL_ANN[args.dataset]
        exp.train_ann = args.train_ann or _TRAIN_ANN[args.dataset]

        if args.task is not None:
            assert (
                args.task in _SUPPORTED_TASKS[args.dataset]
            ), "The specified task cannot be performed with the given dataset!"
            if args.dataset == "lmo" or args.dataset == "lm":
                if args.task == "object_pose":
                    exp.object_pose = True
    # exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()

    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir)
        ckpt_file = os.path.join(exp.output_dir, "best_ckpt.pth")
        ckpt_file = ckpt_file if os.path.isfile(ckpt_file) else os.path.join(exp.output_dir, 'latest_ckpt.pth')
    elif args.ckpt == "random":
        pass
    else:
        ckpt_file = args.ckpt

    if args.ckpt == "random":
        #Proceed with initialized values
        ckpt = None
    else:
        # load the model state dict
        ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if ckpt is not None:
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    if not args.export_det:
        model.head.decode_in_inference = False
    if args.export_det:
        if args.task == "object_pose":
            if args.dataset == 'ycbv':
                camera_matrix = model.head.cad_models.camera_matrix['camera_uw']  #camera_matrix for val split
            elif args.dataset == 'lmo' or args.dataset == "lm":
                camera_matrix = model.head.cad_models.camera_matrix
            post_process = PostprocessExport(conf_thre=0.4, nms_thre=0.01, num_classes=exp.num_classes, object_pose=True, camera_matrix=camera_matrix)
        elif args.task == "human_pose":
            post_process = PostprocessExport(conf_thre=0.05, nms_thre=0.45, num_classes=exp.num_classes, task=args.task)
        else:
            post_process = PostprocessExport(conf_thre=0.25, nms_thre=0.45, num_classes=exp.num_classes)
        model_det = nn.Sequential(model, post_process)
        model_det.eval()
        args.output = 'detections'

    logger.info("loading checkpoint done.")
    if args.dataset == 'ycbv':
        img = cv2.imread("../edgeai-yolox/assets/ti_mustard.png")
    elif args.dataset == 'lmo':
        img = cv2.imread("../edgeai-yolox/assets/sample_lmo_pbr.jpg")
    elif args.dataset == "tree_kpts":
        img = cv2.imread("../edgeai-yolox/assets/tree_kpts_sample.png.png")
    else:
        img = cv2.imread("../edgeai-yolox/assets/dog.jpg")
    img, ratio = preprocess(img, exp.test_size)
    img = img[None, ...]
    img = img.astype('float32')
    img = torch.from_numpy(img)
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    if args.export_det:
        output = model_det(img)

    if args.export_det:
        torch.onnx.export(
            model_det,
            img,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
        )
        logger.info("generated onnx model named {}".format(args.output_name))
    else:
        torch.onnx._export(
            model,
            img,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
        )
        logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))
    else:
        import onnx
        onnx.shape_inference.infer_shapes_path(args.output_name, args.output_name)

def run_export(**kwargs):
    logger.info("kwargs value: {}".format(kwargs))
    exp = get_exp(None, kwargs['name'])
    exp.max_epoch = kwargs['max_epoch']
    exp.warmup_epochs = max(min(exp.warmup_epochs, exp.max_epoch//4), 1)
    exp.no_aug_epochs = max(min(exp.no_aug_epochs, exp.max_epoch//4), 1)
    exp.output_dir = kwargs['output_dir']
    with open(kwargs['train_ann']) as train_ann_fp:
        train_anno = json.load(train_ann_fp)
        categories = train_anno['categories']
        exp.num_kpts = len(categories[0]['keypoints'])

    main(kwargs=kwargs, exp=exp)


if __name__ == "__main__":
    main()
