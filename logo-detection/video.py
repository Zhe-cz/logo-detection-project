from keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import Label

yolo_net = YOLO()

capture_frame = cv2.VideoCapture("img/manynike.mp4")
fourcc_format = cv2.VideoWriter_fourcc(*'XVID')
output_frame = cv2.VideoWriter('output.avi',fourcc_format, 30, (600,600))
frames_per_second = 0.0
while (True):
    time1 = time.time()

    ref, frame = capture_frame.read()
    if ref == True:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(np.uint8(frame))
        im, img = yolo_net.detecter_images(frame)
        frame = np.array(img)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame,(600,600))
        frames_per_second = (frames_per_second + (1. / (time.time() - time1))) / 2
        print("frames_per_second= %.2f" % (frames_per_second))
        frame = cv2.putText(frame, "frames_per_second= %.2f" % (frames_per_second), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 244), 1)

        output_frame.write(frame)
        cv2.imshow("aaaa", frame)
        keys = cv2.waitKey(1) & 0xff
        if keys == 27:
            capture_frame.release()
            break
    else:
        break
capture_frame.release()
yolo_net.close_session()
cv2.destroyAllWindows()
brands_list = (yolo_net.band_dict[k]//30 for k in yolo_net.band_dict.keys())
brands_list = list(brands_list)
window = tk.Tk()
window.title('Times')
window.geometry('400x600')
Label(text='361:{}\n'
           'adidas:{}\n'
           'anta:{}\n'
           'erke:{}\n'
           'kappa:{}\n'
           'lining:{}\n'
           'nb:{}\n'
           'nike:{}\n'
           'puma:{}\n'
           'xtep:{}\n'.format(brands_list[0],brands_list[1],brands_list[2],brands_list[3],brands_list[4],brands_list[5],brands_list[6],brands_list[7],brands_list[8],brands_list[9])
      ,font=('宋体',30)).pack()
tk.mainloop()
