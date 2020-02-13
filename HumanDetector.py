
# coding: utf-8
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import telegram_send
import time
from datetime import datetime

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

########## PORTION TO BE CONFIGURED ################ 
IP = "10.0.x.x"
username = "admin"
password = "password"
frameSkipped = 10 # Analyse a frame every X
frameScaling = 50 # Image resize ratio, in percentage
confidence = 0.6  # Consider object detected if confidence is more than this value
waitOnDetection = 30 # Seconds
###################################################

# RTSP URL construction (DAHUA CAMS)
streamURL = "rtsp://"+ username + ":" + password + "@" + IP + ":554/cam/realmonitor?channel=1&subtype=0"

# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()

  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Open stream
camStream = cv2.VideoCapture(streamURL)

print(str(datetime.now()) + " - " + IP + " - Processing started")
with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while True:
       try:
          # only process one in X frames
          for i in range(1,frameSkipped):    
          
                # Capture frame
                camStream.grab()
        
          # Get frame
          ret, frame = camStream.read()
      
       except:
          print(str(datetime.now()) + " - " + IP + " - Error getting snapshot")
          time.sleep(5)
          
          # Empty framebuffer and reconnect
          camStream.release()
          camStream = cv2.VideoCapture(streamURL)
          
       else:
          # Change colorspace 
         image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
         # Scale frame if needed
         if frameScaling < 100:
             width = int(image_np.shape[1] * frameScaling / 100)
             height = int(image_np.shape[0] * frameScaling / 100)
             dim = (width, height)
             # resize image
             image_np = cv2.resize(image_np, dim, interpolation = cv2.INTER_AREA)
      
         # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
         image_np_expanded = np.expand_dims(image_np, axis=0)
         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      
         # Each box represents a part of the image where a particular object was detected.
         boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      
         # Each score represent how level of confidence for each of the objects.
         # Score is shown on the result image, together with the class label.
         scores = detection_graph.get_tensor_by_name('detection_scores:0')
         classes = detection_graph.get_tensor_by_name('detection_classes:0')
         num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      
         # Actual detection.
         (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
         # ONLY SHOW ONE CLASS (1 = people)
         boxes = np.squeeze(boxes)
         scores = np.squeeze(scores)
         classes = np.squeeze(classes)
         indices = ((classes == 1) & (scores > confidence)).nonzero()[0]
 
         # In case objects detected, send message
         if len(indices) > 0:

             print(str(datetime.now()) + " - " + IP + "- People detected: " + str(len(indices)))

             boxes = boxes[indices]          
             scores = np.squeeze(scores[indices])
             classes = np.squeeze(classes[indices]) 

             # Visualization of the results of a detection
             vis_util.draw_bounding_boxes_on_image_array(image_np,boxes)

             # Save image for telegram... TODO: is it necessary??
             cv2.imwrite(IP + ".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

             with open(IP + ".jpg", "rb") as f:
                  telegram_send.send(images=[f], captions=["ALERT: Intruder detected: " + str(len(indices))])
         
             # Wait a bit, not to flood telegram
             time.sleep(waitOnDetection)
            
             # Empty framebuffer and reconnect
             camStream.release()
             camStream = cv2.VideoCapture(streamURL)
            





