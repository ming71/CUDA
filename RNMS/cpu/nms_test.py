import torch
import numpy as np
import cv2

from nms_wrapper import nms

# input xyxya!
def get_rotated_coors(box):
    try:
        fbox = box.clone()
    except:
        fbox = box.copy()
    fbox[0] = (box[2]+box[0])*0.5
    fbox[1] = (box[2]+box[0])*0.5
    fbox[2] = box[2]-box[0]
    fbox[3] = box[3]-box[1]
    box = fbox
    assert len(box) > 0 , 'Input valid box!'
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-a, center=(cx,cy), scale=1) # anti-clkwise
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 

    if isinstance(x0,torch.Tensor):
        r_box=torch.cat([x0.unsqueeze(0),y0.unsqueeze(0),
                         x1.unsqueeze(0),y1.unsqueeze(0),
                         x2.unsqueeze(0),y2.unsqueeze(0),
                         x3.unsqueeze(0),y3.unsqueeze(0)], 0)
    else:
        r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
    return r_box



if __name__ == '__main__':
    boxes = np.array([[110, 110, 210, 210, 0,       0.88],
                      [100, 100, 200, 200, 0,       0.99],  # res1
                      [100, 100, 200, 200, 10,     0.66],
                      [250, 250, 350, 350, 0.,      0.77]],  # res2
                      dtype=np.float32)
    
    dets_th=torch.from_numpy(boxes).cuda()
    iou_thr = 0.1
    print(dets_th.shape)
    inds = nms(dets_th, iou_thr)
    print(inds)     
    
    img = np.zeros((1000,1000,3), np.uint8)
    img.fill(255)
    
    boxes = boxes[:, :-1]
    cbox = (255,0,0)    # format GBR!!
    ctar = (0,0,255)    # red is target!!
    boxes = [get_rotated_coors(i).reshape(-1,2).astype(np.int32)  for i in boxes]
    for idx, box in enumerate(boxes):
        color = ctar if  idx in inds else cbox
        img = cv2.polylines(img,[box],True,color,1)
        cv2.imshow('anchor_show', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()             
                      
                      
                      
                      
                      
