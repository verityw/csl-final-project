import argparse
import base64
from datetime import datetime
import os
import shutil
import csv

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import model as m

from tensorflow.keras.models import load_model
import tensorflow as tf

import utils

sio = socketio.Server()
app = Flask(__name__)
model = None
# prev_image_array = None # This was in the original code. Don't know what it does. Gonna use the variable for storing previous images when recurrent.
MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # print(args.im_list)
        # save frame
        if args.image_folder != '':
            #timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            #image_filename = os.path.join(args.image_folder, timestamp)
            #image.save('{}.jpg'.format(image_filename))
            pass
            
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array
            args.im_list.append(image)
            if len(args.im_list) > utils.SEQUENCE_LEN:
                args.im_list = args.im_list[-utils.SEQUENCE_LEN:]
            # add image to buffer
            # TO DO
            steering_angle = 0
            if len(args.im_list) == utils.SEQUENCE_LEN:
                imgs = np.concatenate(args.im_list).reshape([1, 16, 160, 320, 3])
                # print(imgs.shape)
                steering_angle = model.predict(imgs, batch_size=1)[0,0,-1]
                # print("HELLO", steering_angle)
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            if len(args.im_list) < utils.SEQUENCE_LEN:
                throttle = 0
            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
            if args.image_folder != "":
                csv_name = os.path.join(args.image_folder, args.model[:-3] + ".csv")
                with open(csv_name, mode='a') as telemetry_csv:
                    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                    telemetry_writer = csv.writer(telemetry_csv, delimiter = ',', quotechar = '"')
                    telemetry_writer.writerow([timestamp, steering_angle, throttle, speed]) 

        except Exception as e:
            print(e)
        
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-i', help='list of images fake arg',  dest='im_list',         type=list, default=[])
    # parser.add_argument(
    #     'is_rnn',
    #     type=bool,
    #     nargs='?',
    #     help='Whether or not the loaded model is an RNN'
    # )
    args = parser.parse_args()
    model_builder = m.build_rnn
    model = model_builder(args)
    print(args.model)
    model.load_weights(args.model)
    model.summary()
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
