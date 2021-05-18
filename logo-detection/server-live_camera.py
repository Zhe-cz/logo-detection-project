import tornado
from tornado.options import define, options
import tornado.ioloop
import tornado.options
import tornado.httpserver
import tornado.web
import os, json
from yolo import YOLO
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
import time

yolo = YOLO()

from flask import Flask, render_template, Response
app = Flask(__name__)



app = Flask(__name__)
camera = cv2.VideoCapture(0) 

def gen_frames():  
    fps = 0.0
    while True:
        t1 = time.time()
        success, frame = camera.read() 
        if not success:
            break
        else:
            frame = Image.fromarray(np.uint8(frame))
            img1, image = yolo.detecter_images(frame)
            frame = np.array(image)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index11.html')


if __name__ == '__main__':
    app.run(debug=True)



