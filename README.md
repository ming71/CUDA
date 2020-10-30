# codebase
Useful cuda codes with annotations inside . 

## RoIAlign
### roi_align  
I've heavily commented the code about RoIAlign for better usage . Two different version are included , and no much different effect founded between them.   

### rotate_roi_align  
(Not checked yet!) RoIAlign for rotated rois . Not recommended before my checkout .

## Rotated-NMS
Code for skew-bboxes NMS calculation.
GPU and CPU are all supported now. 

## DCNv2
Replaceable and faster implemention for conv.Directly apply to replace existed conv layer.   
**bug**: not support apex mix precision training or half, in debuging.

## ORN
Rotated conv layer.  
Test demo also included for MNIST.(I've fixed a bug and made it supportable for pytorch with version higher than 1.0.0.)

## point_justify

Tested on pytorch 1.1+ with CUDA 9.0. The code is helpful to achieve ATSS with rotation bound boxes.