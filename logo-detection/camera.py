from keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
yolo_net = YOLO()

capture_frame=cv2.VideoCapture(0)

frames_per_second = 0.0
while(True):
    time1 = time.time()
    ref, frame = capture_frame.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))

    im, img = yolo_net.detecter_images(frame)
    frame = np.asarray(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    frames_per_second = (frames_per_second + (1./(time.time()-time1)) ) / 2
    print("frames_per_second= %.2f"%(frames_per_second))
    frame = cv2.putText(frame, "frames_per_second= %.2f"%(frames_per_second), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    
    cv2.imshow("aaaa",frame)
    keys = cv2.waitKey(0) & 0xff
    if keys == 12:
        capture_frame.release()
        break

yolo_net.close_session()
    
