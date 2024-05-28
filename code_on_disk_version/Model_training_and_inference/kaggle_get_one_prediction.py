#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import sys

from processor.recognition import REC_Processor

# make sure to provide correct paths to the folders on your machine


#video_path = '20221012_602_1_washstep6_repeat0.mp4'
video_path = '20221014_670_1.mp4'
fps = 25
wind_leng = 25
image_size=(1080, 1920)
interpolation='bilinear'
num_channels = 3
saved_skelton_video = "video_with_keypoints.mp4"
saved_wash_step_video = "video_with_wash_step.mp4"


num_hand_in = 4
num_hand_out = 2
key_point_number = 21

# define variables for hand key points
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_hand_key_point(IMAGE_FILES):
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

def split_and_prep_frame(video_path):
    vid_cap = cv2.VideoCapture(video_path)

    is_success = True
    frame_num = 0
    keypoint_list = []
    # save video with hand key points
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    videoWrite = cv2.VideoWriter(saved_skelton_video, fourcc, 25, (1920,1080))
    
    while is_success:
        is_success, image = vid_cap.read()
        
        if is_success:
            image, key_points = get_hand_key_point(image)
            
            #generate video
            videoWrite.write(image)
            #collect key points
            keypoint_list.append(key_points)
            frame_num += 1
            
    videoWrite.release() #finish generating hand key points
    keypoint_tensor = np.stack(keypoint_list, axis = 1) #concate key point array along time axis
    keypoint_input = []
    for i in range(frame_num-wind_leng+1):
        keypoint_sample = keypoint_tensor[:, i:i+wind_leng]

        keypoint_input.append(keypoint_sample[np.newaxis,:])

    return keypoint_input

def map_result_to_step(action_result, frame_step):
    action_list = []
    action_list.extend(["step: unknown"]*frame_step)
    for action_type in action_result:
        label = np.argmax(action_type)
        if label == 0:
            action_list.append("step: "+str(7))
        else:
            action_list.append("step: "+str(label))
    print(len(action_list))
    return action_list


def add_label_to_video(saved_skelton_video, saved_wash_step_video, action_list, fps):
    vid_cap = cv2.VideoCapture(saved_skelton_video)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    width, height = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWrite = cv2.VideoWriter(saved_wash_step_video, fourcc, fps, (width, height))
    frame_num = 0
    is_success, image = vid_cap.read()  
    while is_success:    
        cv2.putText(image, action_list[frame_num], (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video',image)
        videoWrite.write(image)
        is_success, image = vid_cap.read()
        frame_num += 1
    vid_cap.release()
    videoWrite.release()



# get key point information and video with skeleton
keypoint_input = split_and_prep_frame(video_path)
print("keypoint_input",len(keypoint_input), keypoint_input[0].shape)

# load model and inference with the loaded model
processor = REC_Processor(sys.argv[1:])
processor.start(keypoint_input)

# generate list for detected action
print("processor.result", processor.result[0].shape)

action_list = map_result_to_step(processor.result, wind_leng-1)
print("action_list", processor.result)
# add action to the video
add_label_to_video(saved_skelton_video, saved_wash_step_video, action_list, fps)

