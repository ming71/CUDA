## Rotated NMS calculation
### usage
* **install**  
```python
bash build.sh
```
* **call**  
pred : (num_boxes, 6)  
6 = x y w h t conf
(no need to asort before input)  
transform to torch.Tensor first, more details can be found in test file. 
```python
import r_nms
inds = r_nms.r_nms(pred, iou_thr)
```
### test
Run `nms_test.py` to vis and return the test result.

### attention
1. Import torch before `r_nms`, or you can  reinstall pytorch(really stupid method)
2. Input with shape  (num_boxes, 6), however length of each box> 6 is also ok, but only running without bugs, validity is not sured.(In fact, deprecated.I don't konw where the question lies.) 
3. thetas present in **radians measure**.
4. thetas start from 0 (x+), and **anclockwise** is positive.
5. In test file, theta in cv2.getRotationMatrix2D ought to be anticlockwise.
