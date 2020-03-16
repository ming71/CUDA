## Rotated NMS calculation
### usage
* **install**  
```python
bash build.sh
```
* **call**  
bbox : (num_boxes, 6)  
6 = **x y x y t conf**
(no need to asort before input)  
more details can be found in test file. 

### test
Run `nms_test.py` to vis and return the test result.

### attention
1. thetas present in **degree measure**.
2. theta ranges from -90 to 90, and clockwise is positive.
3. In test file, theta in cv2.getRotationMatrix2D ought to be anticlockwise, thus `-angle` taken as input.

