# HumanDetector
Simple script to detect humans on [Dahua] surveillance cameras, using Tensorflow object detection 

PREREQUISITES:
sudo apt install python3-pip
pip3 install tensorflow
pip3 install matplotlib
pip3 install opencv-python
pip3 install telegram-send
pip3 install pillow
pip3 install --upgrade protobuf
sudo apt  install protobuf-compiler
git clone https://github.com/tensorflow/models 

PREPARE
cd model/research
protoc object_detection/protos/*.proto --python_out=.
cd model/research/object_detection
cp <source>/humanDetector.py humanDetector.py 

RUN
python3 humanDetector.py 
