import argparse
import os
import sys
import os.path as osp

import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer_web

import streamlit as st
import cv2
import time
from PIL import Image 
import numpy as np


@torch.no_grad()
def run(weights=osp.join("./", 'yolov6s.pt'),
        source=osp.join("./", 'test.jpg'),
        yaml='./data/coco.yaml',
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        save_txt=False,
        save_img=True,
        classes=None,
        agnostic_nms=True,
        project=osp.join("./", 'runs/inference'),
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False,
        img_source = None
        ):
    
    # create save dir
    save_dir = osp.join(project, name)
    if (save_img or save_txt) and not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        LOGGER.warning('Save directory already existed')
    if save_txt:
        os.mkdir(osp.join(save_dir, 'labels'))

    # Inference
    inferer = Inferer_web(source, weights, device, yaml, img_size, half, img_source)
    ret_img = inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf)
    return ret_img
    # if save_txt or save_img:
    #     LOGGER.info(f"Results saved to {save_dir}")




selection = st.sidebar.radio("Demo YoloV6", ["Home", "Detection"])

if selection == "Home":
    st.title("YOLOv6 Object Detection Demo")
    st.subheader("Creator - Kaushal R.")

if selection == "Detection":
    st.title("YOLOv6 Object Detection Demo")

    available_model = [
        'yolov6', 'resnet50', 'yolov4', 
    ]

    #st.radio("colour", ['r', 'g', 'b'], index=1)
    model_select = st.selectbox("Select Model", available_model, index=0)

    if model_select == "yolov6":

        score_thres = st.slider("conf", min_value = 0, max_value = 100, value = 45)
        #st.number_input("conf_number", min_value = 0, max_value = 100, value = 90)

        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

        if image_file is not None:

            original_image = Image.open(image_file)
            ret_img = run(source=image_file.name, conf_thres = score_thres/100, img_source=original_image)
            original_image = np.array(original_image)

            st.write("**Detections**")
            ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
            ret_img = Image.fromarray(ret_img)
            st.image(ret_img)

            st.write("Original Image")
            st.image(original_image)
    
    else:

        st.subheader("Coming Soon !!")

    


if selection == "classification":
    st.title("Object classification")
    st.subheader("Coming Soon !!")
    # st.subtitle("under construction")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.001)
        progress.progress(i+1)
    st.balloons()

