#!/usr/bin/env python3
import logging
import os

main_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - \n%(message)s\n', 
                                   datefmt='%d-%m-%Y %H:%M:%S')
bl_formatter = logging.Formatter('%(message)s')

#Log
def save_log(message):
    fh = logging.FileHandler(os.path.join(r"/home/ubuntu/handwash/new-ai-handwash-server-main", "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(bl_formatter)
    logger = logging.getLogger('Python data')
    logger.addHandler(fh)
    logger.info("**************************************")

    fh.setFormatter(main_formatter)
    logger.addHandler(fh)
    logger.exception(message)

    fh.setFormatter(bl_formatter)
    logger.addHandler(fh)
    logger.info("**************************************\n\n")

from datetime import datetime
import json
import sys
from pprint import pprint

import cv2
import eventlet
import mediapipe as mp
import numpy as np
import socketio
from aiohttp import web
import aiohttp_cors
from scipy.special import softmax

from processor.recognition import REC_Processor

# make sure to provide correct paths to the folders on your machine   
interpolation='bilinear'
num_channels = 3
num_hand_in = 4
num_hand_out = 2
key_point_number = 21

# define variables for hand key points
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_hand_key_point_original(IMAGE_FILES):
    # For static images:

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.5) as hands:

        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(IMAGE_FILES, 1)
        
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        #set initial value for keypoint
        key_point_numpy = np.zeros((num_channels, key_point_number, num_hand_in))
        score_numpy = np.zeros(num_hand_in) # mofidy the order of skeleton according to the score for each hand

        # Print handedness and draw hand landmarks on the image.
        # print('Handedness:', results.multi_handedness[0].classification[0].score)
        if results.multi_hand_landmarks:
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            
            #record each hand information for each hand 
            for m, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # print('hand_landmarks:', hand_landmarks.landmark[20].x)
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                # )
                score_numpy[m] = results.multi_handedness[m].classification[0].score
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                #record keypoint information for each point
                for key_p in range(21):
                    key_point_numpy[0, key_p, m] = hand_landmarks.landmark[key_p].x
                    key_point_numpy[1, key_p, m] = hand_landmarks.landmark[key_p].y
                    key_point_numpy[2, key_p, m] = hand_landmarks.landmark[key_p].z

            IMAGE_FILES_labeled = cv2.flip(annotated_image, 1)
    
        else:
            IMAGE_FILES_labeled = cv2.flip(image, 1)

        # get hands with largest score
        sort_index = (-score_numpy).argsort()

        key_point_numpy[:, :, :] = key_point_numpy[:, :, sort_index] #not understand transpose C*T*V*M to V*T*M*C
    
        key_point_numpy = key_point_numpy[:, :, 0:num_hand_out]
        
        #combine the two hands into a graph
        key_point_numpy = np.concatenate((key_point_numpy[:,:,0:1], key_point_numpy[:,:,1:2]), axis=-2)
 
        return IMAGE_FILES_labeled, key_point_numpy
    
def get_hand_key_point(data):
    # For static images:

        key_point_numpy = np.zeros((num_channels, key_point_number, num_hand_in))
        score_numpy = np.zeros(num_hand_in) # mofidy the order of skeleton according to the score for each hand
        
        # Open the JSON file for reading
        # with open('test2.json', 'r') as json_file:
        #     data = json.load(json_file)
            
        for index, side in enumerate(data):
            keypoints3D = data[side][0]["keypoints3D"]
            score_numpy[index] = data[side][0]["score"]
            
            for number, point in enumerate(keypoints3D):
                key_point_numpy[0, number, index] = point['x']
                key_point_numpy[1, number, index] = point['y']
                key_point_numpy[2, number, index] = point['z']

        # for i in data:        
        #     for index, side in enumerate(i):
        #         keypoints3D = i[side][0]["keypoints3D"]
        #         score_numpy[index] = i[side][0]["score"]
                
        #         for number, point in enumerate(keypoints3D):
        #             key_point_numpy[0, number, index] = point['x']
        #             key_point_numpy[1, number, index] = point['y']
        #             key_point_numpy[2, number, index] = point['z']        
    
                
        # get hands with largest score
        sort_index = (-score_numpy).argsort()

        key_point_numpy[:, :, :] = key_point_numpy[:, :, sort_index] #not understand transpose C*T*V*M to V*T*M*C
        
        key_point_numpy = key_point_numpy[:, :, 0:num_hand_out]
        key_point_numpy=np.concatenate((key_point_numpy[:,:,0:1],key_point_numpy[:,:,1:2]),axis=-2)
        return key_point_numpy

def split_and_prep_frame(input):
    keypoint_list = []
    
    for i in input:
        # key_points = get_hand_key_point(i)
        image, key_points = get_hand_key_point_original(i)
        keypoint_list.append(key_points)
        
    keypoint_tensor = np.stack(keypoint_list, axis = 1) 
    
    # concate key point array along time axis
    keypoint_input = []
    
    keypoint_input.append(keypoint_tensor[np.newaxis,:])
    
    # #concate key point array along time axis
    # keypoint_input = []
    # for i in range(25):
    #     keypoint_sample = keypoint_tensor[:, :]
    #     keypoint_input.append(keypoint_sample[np.newaxis,:])

    # print(f"keypoint_input: {keypoint_input}")
    return keypoint_input

# this helper function is to map the result through steps, since the 0 index is step 7
def sort_list(prob_list:list):
    temp=prob_list[0]
    copy=prob_list[1:]
    copy.append(temp)
    return copy

def map_result_to_step(action_result):
    action_list = []
    # action_list.extend(["step: unknown"])
    print(f"length of action_result: {len(action_result)}")
    
    for action_type in action_result:
        label = np.argmax(action_type)
        print(f"action_type[0]: {action_type[0]}")
        # save_log(f"label: {label}")

        prob_list=softmax(action_type).tolist()
        print(f"normalized: {prob_list}")
        sorted_list=sort_list(prob_list[0])
        print(f"sorted: {sorted_list}")

        if label == 0:
            action_list= {
                "step":7,
                "probabilities":sorted_list
            }
        else:
            action_list= {
                "step":label,
                "probabilities":sorted_list
            }
        # action_list=("step: "+str(label+1))
    return action_list


# creates a new Async Socket IO Server
sio = socketio.AsyncServer(cors_allowed_origins='*',async_mode='aiohttp')
# Creates a new Aiohttp Web Application
app = web.Application()
# Binds our Socket.IO server to our Web App instance
sio.attach(app)

cors = aiohttp_cors.setup(app)
# app.router.add_routes(routes) if you have routes
# app.router.add_static("/", rootdir) if you want to serve static, and this has to be absolutely the last route since it's the root. Adding any route after this becomes ignored as '/' matches everthing.
for resource in app.router._resources:
    # Because socket.io already adds cors, if you don't skip socket.io, you get error saying, you've done this already.
    if resource.raw_match("/socket.io/"):
        continue
    cors.add(resource, 
    { 
        '*': aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*") 
    })

# we can define aiohttp endpoints just as we normally
# would with no change
async def index(request):
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

# If we wanted to create a new websocket endpoint,
# use this decorator, passing in the name of the
# event we wish to listen out for
# @sio.on('message')
# async def print_message(sid, message):
def print_message(sid, message):
    # When we receive a new event of type
    # 'message' through a socket.io connection
    # we print the socket ID and the message
    # save_log(f"Socket ID: {sid}")
    # print(f"message: {message}")
    # save_log(type(message))
    # get key point information and video with skeleton
    keypoint_input = split_and_prep_frame(message)
    # print(f"keypoint_input: {keypoint_input}")
    # print(f"keypoint_input shape: {len(keypoint_input)} {keypoint_input[0].shape}")
    
    # load model and inference with the loaded model
    test_list = [keypoint_input]
    processor = REC_Processor(sys.argv[1:])
    processor.start(keypoint_input)

    # save_log(f"processor.result {processor.result}" )
    # print("processor.result", processor.result[0].shape)
    action_list = map_result_to_step(processor.result)
    # print(f"action_list {action_list}")
    print(f"action {action_list['step']}")
    return action_list['step']
    # await sio.emit('message', action_list)
    # save_log(f"Send back message: {action_list}")
# We bind our aiohttp endpoint to our app
# router
app.router.add_get('/', index)

# We kick off our server
if __name__ == '__main__':
    # change the port as the default port number 8080 has been occupied
    # web.run_app(app,port=9500)
    video_path = '/home/ubuntu/handwash/Model_training_and_inference/20221014_670_1.mp4'
    vid_cap = cv2.VideoCapture(video_path)
    is_success = True
    image_queue = []
    step_list = []
    while is_success:
        is_success, image = vid_cap.read()
        image_queue.append(image)
        if len(image_queue) == 25:
            step = print_message(None, image_queue)
            step_list.append(step)
            del image_queue[0]
    print(f"step list {step_list}")


# # get key point information and video with skeleton
# keypoint_input = split_and_prep_frame()

# # load model and inference with the loaded model
# test_list = [keypoint_input]
# processor = REC_Processor(sys.argv[1:])
# processor.start(keypoint_input)

# print("processor.result", processor.result)
# # print("processor.result", processor.result[0].shape)
# action_list = map_result_to_step(processor.result)
# print("action_list", action_list)


