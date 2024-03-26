#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
import time

import torch
import torchvision

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, vis, multiclass_nms, boxes


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default='test_image.png',
        help="Path to your input images directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.8,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="384,672",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save results to a text file.",
    )

    return parser

def cxcywh2xyxy_export(cx,cy,w,h):
    #This function is used while exporting ONNX models
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    halfw = w/2
    halfh = h/2
    xmin = cx - halfw  # top left x
    ymin = cy - halfh  # top left y
    xmax = cx + halfw  # bottom right x
    ymax = cy + halfh  # bottom right y

    return torch.cat((xmin, ymin, xmax, ymax), 2)

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    prediction = torch.from_numpy(prediction)
    cx, cy, w, h = prediction[..., 0:1], prediction[..., 1:2], prediction[..., 2:3], prediction[..., 3:4]
    box = cxcywh2xyxy_export(cx, cy, w, h)
    prediction[:, :, :4] = box

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf =  image_pred[:, 4:5] * class_conf
        conf_mask = (conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        tensor_cat_inp = [image_pred[:, :4], conf, class_pred.float()]
        tensor_cat_inp.extend([image_pred[:,6:]])
        detections = torch.cat(tensor_cat_inp, 1)

        detections = detections[conf_mask]
        class_2d_offset = detections[:, -1:] * 4096  # class_2d_offser

        nms_out_index = torchvision.ops.nms(
            detections[:, :4] + class_2d_offset,
            detections[:, 4],
            nms_thre,
        )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def nms(boxes, scores, threshold):
    """
    Perform non-max suppression on a set of bounding boxes and corresponding scores.

    :param boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
    :param scores: a list of corresponding scores
    :param threshold: the IoU (intersection-over-union) threshold for merging bounding boxes
    :return: a list of indices of the boxes to keep after non-max suppression
    """
    # Sort the boxes by score in descending order
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))

            # A_i + A_j - intersection
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            if iou > threshold:  # Remove boxes with IoU greater than the threshold
                order.remove(j)
    return keep

if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))

    image_files = [f for f in os.listdir(args.images_path) if os.path.isfile(os.path.join(args.images_path, f))]
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Filter for image files

    session = onnxruntime.InferenceSession(args.model)

    total_inference = 0

    # for image_file in image_files:
    if True:
        image_file = "IMG_20240318_124733.png"
        print(os.path.join(args.images_path, image_file))
        origin_img = cv2.imread(os.path.join(args.images_path, image_file))
        img, ratio = preprocess(origin_img, input_shape)

        start = time.perf_counter()
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        total_inference += (time.perf_counter() - start)

        dets = output[0]
        # each row in dets contains the following in order:
        # reg_output (bbox), obj_output, cls_output, kpts_output
        # dets = postprocess(output[0], 1, conf_thre=args.score_thr, nms_thre=0.45)[0]
        # filter for sufficient scores. conf is the objectness * class_conf
        # dets[:, 5] guaranteed class, since only trees :)
        # valid_score = dets[:, 4] * dets[:, 5] > args.score_thr
        # dets = dets[valid_score]

        dets[:, :4] /= ratio # rescale the bbox
        dets[:, 6::3] /= ratio # rescale x of kpts
        dets[:, 7::3] /= ratio # rescale y of kpts

        # net learns coco bbox format: [cx, cy, w, h]
        # we transform it to different format for nms calcs: [xmin, ymin, xmax, ymax]
        # bboxes = boxes.cxcywh2xyxy(dets[:, :4])
        # scores = dets[:, 4] * dets[:, 5]
        # indices = nms(bboxes, scores, 0.1) # merge boxes with IoU > 0.1

        # filter nms results
        # dets = dets[indices]

        for det in dets:
            x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(origin_img, p1, p2, (0,255,0), 2)

            # plot the x and y keypoints with sufficient confidence score
            # conf = objectness * keypoint conf
            for x, y, conf, label in zip(det[6::3], det[7::3], det[4] * det[8::3], ["kpC", "kpL", "kpL", "ax1", "ax2"]):
                if (conf > args.score_thr):
                    # print(f"{label}\t\tx: {x}\ty:{y}\tkptconf:\t{conf}")
                    cv2.circle(origin_img, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        mkdir(args.output_dir)
        output_path = os.path.join(args.output_dir, image_file)
        print(output_path)
        cv2.imwrite(output_path, origin_img)

    print(f"total_inference: {total_inference}, {len(image_files)} images, avg {total_inference / len(image_files)}")