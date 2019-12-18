# codebase
Useful cuda codes with annotations inside . 

## RoIAlign
### roi_align  
I've heavily commented the code about RoIAlign for better usage . Two different version are included , and no much different effect founded between them.   

### rotate_roi_align  
(Not checked yet!) RoIAlign for rotated rois . Not recommended before my checkout .

## Rotated-NMS
Code for skew-bboxes NMS calculation.

## DCNv2
Replaceable and faster implemention for conv.Directly apply to replace existed conv layer. 
**bug**: not support apex mix precision training or half, in debuging.
