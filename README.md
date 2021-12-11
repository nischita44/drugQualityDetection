# DrugQualityDetection

To run the training and inferencing in Google Colab please refer Drug_Quality_Detection.ipynb file provided in this repository.

For complete explaination of this project refer my YouTube Link: 
https://www.youtube.com/watch?v=Q_qWCwX_kMg

**Access Complete Tensorflow 2.4 Model Folder : https://drive.google.com/file/d/10Ramhh92ZqX5gL_fFv4eD3t0K0zAXimJ/view?usp=sharing**

In the above Model Folder route to the below path provided,

\TF2.zip\TF2\models-master\research\object_detection\

In the above path you can find the required folders and files such as inference_graph_12k, images folder( in which you can find train,test data and its corresponding .csv files),train.record and test.record files,test_images_unseen, test_results, xml_to_csv.py, generate_tfrecord.py, model_main_tf2.py,exporter_main_v2.py,Object_detection_image.py.

Final Project: Drug Quality detection using Deep Learning(Neural Networks) and Computer Vision.

Installation- Tensorflow 2.4:

***Cuda installation:***
Ubuntu 18.04 (CUDA 11.0)

Add NVIDIA package repositories

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin 

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub 

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" 

sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb sudo apt-get update

Install NVIDIA driver

sudo apt-get install --no-install-recommends nvidia-driver-450

Reboot. Check that GPUs are visible using the command: nvidia-smi

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb 

sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb sudo apt-get update

**Install development and runtime libraries (~4GB)**

sudo apt-get install --no-install-recommends
cuda-11-0
libcudnn8=8.0.4.30-1+cuda11.0
libcudnn8-dev=8.0.4.30-1+cuda11.0

Install TensorRT. Requires that libcudnn8 is installed above.

sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0
libnvinfer-dev=7.1.3-1+cuda11.0
libnvinfer-plugin7=7.1.3-1+cuda11.0

download cudnn 8 and install in linux base system.

follow the mentioned link for tensorflow 2 Api Installation

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

**How to check whether GPU is available(TRUE)**

In command line terminal run python

import tensorflow as tf

tf.____version____

tf.test.is_gpu_available() --- If GPU is available it should show TRUE.

To check whether training is running on GPU run nvidia-smi

Once you run it you will get the details and you can see the GPU memory consumption

**FOLLOW THE LISTED STEPS FOR TRAINING A MODEL FOR YOUR CUSTOM DATASET USING TENSORFLOW2.4**

Goto TF2 model inside that goto object detection directroy.

create a image folder inside that split your dataset into train and test folder in the ratio of 70: 30 or 75:25 or 80:20 or 90:10

open the terminal from the object detection directory and convert xml to csv : python xml_to_csv.py

Generate tf record for train and test dataset using the below mentioned command.

python generate_tfrecord.py --csv_input=images/train.csv --image_dir=images/train --output_path=train.record

python generate_tfrecord.py --csv_input=images/test.csv --image_dir=images/test --output_path=test.record

train the model:

python model_main_tf2.py --pipeline_config_path=training/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.config --model_dir=training --alsologtostderr

**Evaluate your model**
python Object_detection_image.py --model_dir=faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8 --pipeline_config_path=training/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.config --checkpoint_dir=training

**Export the inference graph:**
python exporter_main_v2.py --trained_checkpoint_dir=training --output_directory=inference_graph_12k --pipeline_config_path=training/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.config

tensorboard:
tensorboard --logdir=training


***DESCRIPTION***

**GOAL**:
Drug Quality Detection at dispatching unit of Drug Manufacturing company to avoid dispatching false quality or badly sealed medicines which might turn out as poison due to bad sealing exposing to air or any pollution or etc.

**APPROACH**: 
Using Deep Learning (Neural Networks) and Computer vision technology for detection of such quality check.

**DATASETS**:
Collected from iPhone with a frame rate of 30fps and collected video data for different variants of drug/ medicine and each video for around 20 seconds. In total there were 10 videos. Then the video data were converted into frames as said before we get 30fps( frame per second ) and we have just considered 4 to 5 frames per second out of 30fps. Once the data was ready I have split the data into train and test where training images were 488 and test images were 122. I used labelImg tool to annotate the images and I have considered 4 classes namely Tablet_Presence, Tablet_Absence, Good_Sealing, Bad_Sealing. This annotation took around 5 hours and the work done was saved in terms of .XML files.

**Stage 1 :Data Collection**

Collected from iPhone with a frame rate of 30fps and collected video data for different variants of drug/ medicine and each video for around 20 seconds. In total there were 10 videos.

**Stage 2: Data Preparation**

Then the video data were converted into frames as said before we get 30fps( frame per second ) and we have just considered 4 to 5 frames per second out of 30fps.

Once the data was ready I have split the data into train and test where training images were 488 and test images were 122 .

**Stage 3: Data Annotation**

I used labelImg tool to annotate the images and I have considered 4 classes namely Tablet_Presence , Tablet_Absence, Good_Sealing, Bad_Sealing . This annotation took around 5 hours and the work done were saved in terms of .XML files .

**Stage 4: Preparing a training model**

Now that we have images and their corresponding.XML files and the tensorflow model accepts the input in terms of .tfrecords so to get that first we need to convert .xml files to .csv and once the conversion is done open the train.csv and test.csv files to proof read if there is any wrong naming of classes are there apart from defined classes .

Post that run the genrate_tfrecord.py file to generate train.record and test.record files.

**Stage 5: Training**

Tensorflow is the framework used and 2.4 is the version .

**Stage 6: Test Results**

![Tablet1_748](https://user-images.githubusercontent.com/20515803/145671713-890436cd-cb90-472a-b607-6ba7b7c9b904.jpg)
![Ointment1_102](https://user-images.githubusercontent.com/20515803/145671720-60f2a0c7-1722-4bc6-b58e-ece1b352c07a.jpg)
![Ointment2_697](https://user-images.githubusercontent.com/20515803/145671722-b16fc2dc-f653-4441-a28d-100eb7b05164.jpg)
![Spray1_51](https://user-images.githubusercontent.com/20515803/145671725-cf8f4744-4fcd-45dc-ba05-935f81d999e2.jpg)
![Spray2_408](https://user-images.githubusercontent.com/20515803/145671729-e472b992-8643-4201-81ff-46a730da426c.jpg)
![Syrup1_102](https://user-images.githubusercontent.com/20515803/145671732-e8ae7fea-11cb-4347-a208-640c6772dc85.jpg)
![Syrup2_153](https://user-images.githubusercontent.com/20515803/145671733-11dd4a6f-d739-4c17-8f42-fc944b59794b.jpg)
![Tablet1_204](https://user-images.githubusercontent.com/20515803/145671737-cd72cb36-3727-4cef-95fb-f9b3b5bd604d.jpg)
![Tablet2_544](https://user-images.githubusercontent.com/20515803/145671740-4b28bee0-356e-4c75-b9c3-0e20cf6fc8e0.jpg)
![Tablet2_1258](https://user-images.githubusercontent.com/20515803/145671744-fcfde1ce-c852-4ee6-b32b-8c4266eedd02.jpg)
![Tablet4_204](https://user-images.githubusercontent.com/20515803/145671752-bc1a0d3c-804e-4653-948c-fd7f129feaf2.jpg)
![Tablet4_340](https://user-images.githubusercontent.com/20515803/145671756-59b0e9ca-3cfc-4282-9cf3-3dd7c0641ba5.jpg)
![Tablet4_1224](https://user-images.githubusercontent.com/20515803/145671758-bbc55ebd-651e-479b-8ce8-edc515baf887.jpg)
![Tablet3_187](https://user-images.githubusercontent.com/20515803/145671810-1d263149-2c71-4901-a121-4697aa5b7b1a.jpg)


