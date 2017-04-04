#!/bin/bash
wget -O ./models/convert_caffemodel/params/vgg_params.npz http://people.eecs.berkeley.edu/~ronghang/projects/cmn/models/vgg_params.npz
wget -O ./models/convert_caffemodel/params/fasterrcnn_vgg_coco_params.npz http://people.eecs.berkeley.edu/~ronghang/projects/cmn/models/fasterrcnn_vgg_coco_params.npz
wget -O ./models/convert_caffemodel/tfmodel/fasterrcnn_vgg_coco_net.tfmodel http://people.eecs.berkeley.edu/~ronghang/projects/cmn/models/fasterrcnn_vgg_coco_net.tfmodel
