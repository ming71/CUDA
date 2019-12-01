## Rotated NMS calculation
### usage
* **install**  
```python
./build.sh
```
* **call**  
pred : (num_boxes, 5)
5= x y w h conf
(no need to asort before input)
```python
import r_nms
inds = r_nms.r_nms(pred, iou_thr)
```
### test
Run `nms_wrapper.py` to vis and return the test result.

### attention
1. Import torch before `r_nms`, or you can  reinstall pytorch(really stupid method)
2. Input with shape  (num_boxes, 5), however length of each box>5 is also ok, but only running without bugs, validity is not sured.(In fact, deprecated.I don't konw where the question lies.) 
3. angle is  in **radians**.