# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# operation
from . import tools

import cv2
import mediapipe as mp
# define variables for hand key points
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_hand_key_point(IMAGE_FILES):
    # Get key point for each frame

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.5) as hands:

        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(IMAGE_FILES, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        # print('Handedness:', results.multi_handedness) # get the label for multiple hands
        hand_info = results.multi_handedness
        
        if results.multi_hand_landmarks:
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            key_point_info = results.multi_hand_landmarks # a list of hand_landmark_list 
            for hand_landmarks in results.multi_hand_landmarks:
                # print('hand_landmarks:', hand_landmarks) # get the label for multiple key points in each hand
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                # )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            IMAGE_FILES_labeled = cv2.flip(annotated_image, 1)

        else:
            IMAGE_FILES_labeled = cv2.flip(image, 1)
            key_point_info = None
    
        # Draw hand world landmarks.
        # if not results.multi_hand_world_landmarks:
        #     continue
        
        return hand_info, key_point_info
    
def process_hand_video(video_path):
    # add key point to video
    vid_cap = cv2.VideoCapture(video_path)

    is_success = True
    frame_num = 0
    hand_info_list = []
    key_point_info_list = []

    # save video with hand key points
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # videoWrite = cv2.VideoWriter("video_with_keypoints.mp4", fourcc, 25, (1920,1080))
    
    while is_success:
        is_success, image = vid_cap.read()
        
        if is_success:
            hand_info, key_point_info = get_hand_key_point(image)
            hand_info_list.append(hand_info)
            key_point_info_list.append(key_point_info)
            # videoWrite.write(image)            
            frame_num += 1
            
    # videoWrite.release()      

    return hand_info_list, key_point_info_list, frame_num


class Feeder_kinetics(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """
    def __init__(self,
                 data_path,
                 # label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=2,
                 num_person_out=1,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        # self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()
        self.split_with_window()

    def load_data(self):
        # load file list
        sample_name_list = []
        sample_label_list = []
        for i in range(7):
            sample_name = os.listdir(os.path.join(self.data_path, str(i)))#[:200]
            sampel_num = len(sample_name)
            sample_name_list.extend(sample_name)
            sample_label_list.extend([i]*sampel_num)
        
        self.sample_name = sample_name_list
        self.label = sample_label_list

        if self.debug:
            self.sample_name = self.sample_name[0:2]
    
        
        # load label
        # label_path = self.label_path
        # with open(label_path) as f:
        #     label_info = json.load(f)   
            
        # sample_id = [name.split('.')[0] for name in self.sample_name]
        # self.label = np.array(
        #     [label_info[id]['label_index'] for id in sample_id])
        # has_skeleton = np.array(
        #     [label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        # if self.ignore_empty_sample:
        #     self.sample_name = [
        #         s for h, s in zip(has_skeleton, self.sample_name) if h
        #     ]
        #     self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 3  #channel: the first 3 channels are x, y, z coordinates of hand, the last one is the score
        self.T = self.window_size  #frame
        self.V = 21  #joint: each hand has 21 points
        self.M = self.num_person_out  #person

    def split_with_window(self):
        total_wind_num = 0
        for i, label in enumerate(self.label):
            video_path = os.path.join(self.data_path, str(label), self.sample_name[i]) #get the number of windows for each video segment
            vid_cap = cv2.VideoCapture(video_path)
            totalframecount= int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            wind_num = totalframecount//self.window_size
            total_wind_num += wind_num
        self.window_num = total_wind_num
            
    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        num_hand_in = self.num_person_in*2
        num_hand_out = self.num_person_out*2
        
        sample_name = self.sample_name[index]
        label = self.label[index]
        print("preprocessing wash step", label, "video:", sample_name)
        sample_path = os.path.join(self.data_path, str(label), sample_name)
        
        # get the hand key point data from each video, including coordinate of keypoint, score, label
        hand_info_list, key_point_info_list, frame_num = process_hand_video(sample_path)

        # fill data_numpy
        data_numpy = np.zeros((self.C, frame_num, self.V, num_hand_in))
        score_numpy = np.zeros((frame_num, num_hand_in))
        for frame_index, frame_info in enumerate(hand_info_list):
            if not frame_info: #if there is no hand in one frame, skip the iteration 
                continue
            for m, skeleton_info in enumerate(key_point_info_list[frame_index]):
                #get the score number for each hand in each frame
                score_numpy[frame_index, m] = frame_info[m].classification[0].score
                
                if m >= num_hand_in: #each person has two hands
                    break
                for key_p in range(21):
                    data_numpy[0, frame_index, key_p, m] = skeleton_info.landmark[key_p].x
                    data_numpy[1, frame_index, key_p, m] = skeleton_info.landmark[key_p].y
                    data_numpy[2, frame_index, key_p, m] = skeleton_info.landmark[key_p].z
  

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5 #z axis coordinate use 0 as mean value already
        # data_numpy[0][data_numpy[2] == 0] = 0
        # data_numpy[1][data_numpy[2] == 0] = 0


        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # sort by score according to the sum of key point score for each person
        # sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1) #sort each row
        sort_index = (-score_numpy).argsort(axis=1)
        for t, s in enumerate(sort_index):

            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose(
                (1, 2, 0)) 

        data_numpy = data_numpy[:, :, :, 0:num_hand_out]

        # match poses between 2 frames
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)
        
        #combine the two hands into a graph
        data_numpy = np.concatenate((data_numpy[:,:,:,0:1], data_numpy[:,:,:,1:2]), axis=-2)

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)
