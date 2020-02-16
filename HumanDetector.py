
# coding: utf-8
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models

########## CONFIGURATION ################ 
IP = "10.0.x.x"
username = "admin"
password = "password"
frameSkipped = 10 # Analyse a frame every X
frameScaling = 50 # Image resize ratio, in percentage
confidence = 0.6  # Consider object detected if confidence is more than this value
waitOnDetection = 30 # Seconds
#Active interval. NOTE: If start and stop are equal, means continuous operation
startTime = "21:00"
stopTime = "8:00"
###################################################

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import telegram_send
import time
import datetime

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# helper Code
ARBITRARY_DATE = datetime.datetime(1999, 9, 9)
def is_time_between(t, start, end):
    if start == end:
        return True
    day_add = 1 if end < start else 0
    end_add = 1 if day_add and end == datetime.time(0,0,0,0) else 0
    test_add = 1 if day_add and t < start else 0
    td_time_start = datetime.timedelta(hours=start.hour,
                              minutes=start.minute,
                              seconds=start.second,
                              microseconds=start.microsecond)
    td_time_end = datetime.timedelta(days=day_add + end_add,
                            hours=end.hour,
                            minutes=end.minute,
                            seconds=end.second,
                            microseconds=end.microsecond)
    td_testing = datetime.timedelta(days=test_add,
                           hours=t.hour,
                           minutes=t.minute,
                           seconds=t.second,
                           microseconds=t.microsecond)
    start_date = ARBITRARY_DATE + td_time_start
    end_date = ARBITRARY_DATE + td_time_end
    testing_date = ARBITRARY_DATE + td_testing
    return start_date <= testing_date and testing_date <= end_date

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

# RTSP URL construction (DAHUA CAMS)
streamURL = "rtsp://"+ username + ":" + password + "@" + IP + ":554/cam/realmonitor?channel=1&subtype=0"

# Open CAM stream
camStream = cv2.VideoCapture(streamURL)
if not camStream.isOpened():
    print(str(datetime.datetime.now()) + " - " + IP + " - ERROR: Could not connect to CAM stream, aborting")    
    quit()

print(str(datetime.datetime.now()) + " - " + IP + " - Processing started")
with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while True:
    
       # Are we now in the defined time interval?
       if (startTime != stopTime) and (not is_time_between(datetime.datetime.now().time(),
                              datetime.datetime.strptime(startTime, "%H:%M").time(),
                              datetime.datetime.strptime(stopTime, "%H:%M").time())):
       
           # Not now, sleep a bit and check again
           time.sleep(30)
      
       # Good time, detect 
       else:
           try:
              # only process one in X frames
              for i in range(1,frameSkipped):    
          
                    # Capture frame
                    camStream.grab()

              startCycle = int(round(time.time() * 1000))
        
              # Get frame
              success, frame = camStream.read()
          
              # Check if we got a frame 
              if not success:  
                  print(str(datetime.datetime.now()) + " - " + IP + " - ERROR: CAM did not return a valid frame, reconnecting")
                  raise
           except:
              print(str(datetime.datetime.now()) + " - " + IP + " - Error getting snapshot")
              time.sleep(5)
          
              # Empty framebuffer and reconnect
              camStream.release()
              camStream = cv2.VideoCapture(streamURL)
          
           else:
      
             # Scale frame if needed
             if frameScaling < 100:
                 width = int(frame.shape[1] * frameScaling / 100)
                 height = int(frame.shape[0] * frameScaling / 100)
                 dim = (width, height)
                 # resize image
                 frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
      
             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
             frame_expanded = np.expand_dims(frame, axis=0)
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
              feed_dict={image_tensor: frame_expanded})
      
             # ONLY SHOW ONE CLASS (1 = people)
             boxes = np.squeeze(boxes)
             scores = np.squeeze(scores)
             classes = np.squeeze(classes)
             indices = ((classes == 1) & (scores > confidence)).nonzero()[0]
 
             print("Frame processing Completed in " + str(int(round(time.time() * 1000)) - startCycle) + "ms")
 
             # In case objects detected, send message
             if len(indices) > 0:

                 print(str(datetime.datetime.now()) + " - " + IP + "- People detected: " + str(len(indices)))

                 boxes = boxes[indices]          
                 scores = scores[indices]
                 classes = classes[indices]

                 # Visualization of the results of a detection
                 vis_util.visualize_boxes_and_labels_on_image_array(
                  frame,
                  boxes,
                  classes.astype(np.int32),
                  scores,
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=3) 

                 # Save image for telegram... TODO: is it necessary??
                 cv2.imwrite(IP + ".jpg", frame)

                 with open(IP + ".jpg", "rb") as f:
                      telegram_send.send(conf="/etc/telegram-send.conf", images=[f], captions=["ALERT: Intruder detected: " + str(len(indices))])
             
                 # Wait a bit, not to flood telegram
                 time.sleep(waitOnDetection)
            
                 # Empty framebuffer and reconnect
                 camStream.release()
                 camStream = cv2.VideoCapture(streamURL)
            
