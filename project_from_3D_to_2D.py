# Description: This script is used to project 3D points to 2D image plane

import numpy as np
import os
import cv2
import pickle
from read_write_model import read_model
import pandas as pd
import torch
from scipy.stats import multivariate_normal
import time
from utils.valid_interactions import valid_interactions, colormap_interactions

class Reproject_data():
    def __init__(self):
        #Initialize directories
        self.verbs_EP100_csv = '.../EPIC_100_verb_classes.csv'
        self.verbs_EP100_csv = pd.read_csv(self.verbs_EP100_csv)
        self.sequence_dir = '.../P04_EPIC_55'
        self.labels_dir = os.path.join(self.sequence_dir, '3D_output')
        self.colmap_poses = os.path.join(self.sequence_dir, 'colmap')
        self.imgs_dir = os.path.join(self.sequence_dir, 'selected_plus_guided_rgb')
        self.output_to_show = os.path.join(self.sequence_dir, 'output_to_show')
        if not os.path.exists(self.output_to_show):
            os.makedirs(self.output_to_show)
        self.output_labels_2d = os.path.join(self.sequence_dir, '2d_output_labels')
        if not os.path.exists(self.output_labels_2d):
            os.makedirs(self.output_labels_2d)
        self.output_clusters = os.path.join(self.sequence_dir, 'output_clusters_v2')
        if not os.path.exists(self.output_clusters):
            os.makedirs(self.output_clusters)

        #35 valid verbs
        self.valid_interactions = ['take', 'remove', 'put', 'insert', 'throw', 'wash', 'dry', 'open', 'turn-on', 
                                   'close', 'turn-off', 'mix', 'fill', 'add', 'cut', 'peel', 'empty', 
                                   'shake', 'squeeze', 'press', 'cook', 'move', 'adjust', 'eat', 
                                   'drink', 'apply', 'sprinkle', 'fold', 'sort', 'clean', 'slice', 'pick']
        self.valid_interactions_2 = valid_interactions

        self.colormap_interactions = colormap_interactions
        

        #Read the intrinsic parameters of the sequence
        self.cameras_Colmap, self.imgs_Colmap, self.pts_Colmap = read_model(self.colmap_poses, ext=".txt")
        self.fx = self.cameras_Colmap[1].params[0]
        self.fy = self.cameras_Colmap[1].params[1]
        self.cx = self.cameras_Colmap[1].params[2]
        self.cy = self.cameras_Colmap[1].params[3]
        self.projection_matrix = self.get_projection_matrix()
        self.height = 480
        self.width = 854
        self.size = 500
        self.gaussian = self.get_gaussian()
        self.read_3D_points()
        print('the length of the 3D points is: ', self.points_coord.shape)

    
    def get_projection_matrix(self):
        # Get the projection matrix
        projection_matrix = np.zeros((3, 4))
        projection_matrix[0, 0] = self.fx
        projection_matrix[1, 1] = self.fy
        projection_matrix[0, 2] = self.cx
        projection_matrix[1, 2] = self.cy
        projection_matrix[2, 2] = 1
        return projection_matrix

    def get_camera_pose(self, data):
        #Get the camera translation matrix 
        t = data['colmap']['t_pos']
        R = data['colmap']['R_pos']  
        t_c = (-R.T @ t).reshape(3,1)
        R_c = R.T
        return t_c, R_c
    
    def remap_verb_EP100(self, data):
        ep_verb_class = data['aff_verb_id']
        remapped = self.verbs_EP100_csv[self.verbs_EP100_csv['id'] == ep_verb_class]
        remapped_verb_str = remapped['key'].values[0]
        remapped_verb_id = remapped['id'].values[0]
        return remapped_verb_str, remapped_verb_id

    def read_3D_points(self):
        points, rgb_points = [], []
        self.verb_str, self.verb_id, self.noun_str, self.noun_id = [], [], [], []
        #Iterate over all the files in the directory sequence_dir
        for root, dirs, files in os.walk(self.labels_dir):
            for file in files:
                if file.endswith('.pkl'):
                    pkl = open(os.path.join(root, file), 'rb') #Open a pickle file
                    data = pickle.load(pkl) #Load the pickle file
                    for i in range(len(data['EGOMETRIC_label']['affordance_labels'])):
                        points.append(data['EGOMETRIC_label']['affordance_labels'][i]['3D_aff_points'])
                        rgb_points.append(data['EGOMETRIC_label']['affordance_labels'][i]['3D_aff_colors'])
                        remap_verb_str, remap_verb_id = self.remap_verb_EP100(data['EGOMETRIC_label']['affordance_labels'][i])
                        self.verb_str.append(remap_verb_str)
                        self.verb_id.append(remap_verb_id)
                        self.noun_str.append(data['EGOMETRIC_label']['affordance_labels'][i]['aff_noun'])
                        self.noun_id.append(data['EGOMETRIC_label']['affordance_labels'][i]['aff_noun_id'])
                    pkl.close()
        
        print(len(points), len(rgb_points), len(self.verb_str), len(self.verb_id), len(self.noun_str), len(self.noun_id))
        self.points_coord = np.concatenate(points, axis=0)
        self.points_rgb = np.concatenate(rgb_points, axis=0)

    def object_detector_gt(self, visor_annot):
        #Static objects which are always present in the scene
        objects_in_scene = ['drawer', 'fridge', 'microwave', 'oven', 'sink', 
                            'hob', 'kettle', 'maker:coffee', 'dishwsher', 
                            'machine:washing', 'floor', 'table', 'rubbish']
        #We add the dynamics objects

        if len(visor_annot) > 0:
            for i in range(len(visor_annot['neutral_objects'])):
                objects_in_scene.append(visor_annot['neutral_objects'][i]['noun'])
            for i in range(len(visor_annot['interacting_objects'])):
                objects_in_scene.append(visor_annot['interacting_objects'][i]['noun'])
        return objects_in_scene

    def reproject_points(self, points_in_camera):
        points_in_camera = np.append(points_in_camera, np.ones((1, points_in_camera.shape[1])), axis=0)
        reprojected_points = np.dot(self.projection_matrix, points_in_camera)
        reprojected_points = reprojected_points / reprojected_points[2]
        return reprojected_points
    
    def filter_reprojected_points(self, reprojected_points, present_objects, img_name):
        # Filter the reprojected points by the localization and the semantic noun
        self.good_reprojected_points, self.good_reprojected_rgb, self.good_verbs, self.good_nouns, self.good_interactions = [], [], [], [], []
        for i in range(reprojected_points.shape[1]):
            point = reprojected_points[:, i]
            if point[0] >= 0 and point[0] <= self.width and point[1] >= 0 and point[1] <= self.height:
                if self.noun_str[i] in present_objects and (self.verb_str[i]) in self.valid_interactions:
                    self.good_reprojected_points.append(point)
                    self.good_reprojected_rgb.append(self.points_rgb[i])
                    self.good_verbs.append(self.verb_str[i])
                    self.good_nouns.append(self.noun_str[i])
                    self.good_interactions.append(self.verb_str[i] + ' ' + self.noun_str[i])
        #Save all in a json file
        img_name = img_name.split('.')[0]
        output_2d = {}
        output_2d['points'] = self.good_reprojected_points
        output_2d['rgb'] = self.good_reprojected_rgb
        output_2d['verbs'] = self.good_verbs
        output_2d['nouns'] = self.good_nouns
        output_2d['verb plus noun'] = self.good_interactions
        output_filename = os.path.join(self.output_labels_2d, img_name +'.pkl')
        with open(output_filename, 'wb') as f:
            pickle.dump(output_2d, f)
        print('we are saving the 2d labels in: ', output_filename, 'with a len', len(self.good_reprojected_points))

    def cluster_interactions(self):
        #Cluster the interactions
        self.interaction_clusters = []
        self.interaction_coordinates = {}
        for i in range(len(self.good_verbs)): #Before good_interactions
            if self.good_verbs[i] not in self.interaction_clusters:
                self.interaction_clusters.append(self.good_verbs[i])
        for i in range(len(self.interaction_clusters)):
            self.interaction_coordinates[self.interaction_clusters[i]] = []
        for i in range(len(self.good_verbs)):
            self.interaction_coordinates[self.good_verbs[i]].append(self.good_reprojected_points[i])
        
    def paint_points(self, img, img_name):
        img_copy = img.copy()
        img_name = img_name.split('.')[0]
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(self.good_reprojected_points)):
            point = self.good_reprojected_points[i]
            rgb_point = self.good_reprojected_rgb[i]
            text = self.good_verbs[i] + ' ' + self.good_nouns[i]
            cv2.circle(img_copy, (int(point[0]), int(point[1])), 3, (int(rgb_point[0]), int(rgb_point[1]), int(rgb_point[2])), -1)
            cv2.putText(img_copy, text, (int(point[0]), int(point[1])), font, 1, (255,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.output_to_show, img_name + '.png'), img_copy)
    
    def paint_clusters(self, img, img_name):
        img_copy = img.copy()
        img_name = img_name.split('.')[0]
        font = cv2.FONT_HERSHEY_PLAIN
        #Draw the hotspots of the clusters
        for i in range(len(self.interaction_clusters)):
            cluster = self.interaction_clusters[i]
            prob_sum = np.zeros((self.width, self.height))
            print(prob_sum.shape, 'que mierdas es ')
            for j in range(len(self.interaction_coordinates[cluster])):
                point = self.interaction_coordinates[cluster][j][0:2].astype(int)
                prob = np.zeros((self.width, self.height))

                if (self.width - point[0]) > self.size // 2:
                    gauss_right = self.size
                    prob_right = point[0] + self.size // 2
                else:
                    gauss_right = self.width - point[0] + self.size // 2
                    prob_right = self.width
                
                if point[0] > self.size // 2:
                    gauss_left = 0
                    prob_left = point[0] - self.size // 2
                else:
                    gauss_left = self.size // 2 - point[0]
                    prob_left = 0
                
                if (self.height - point[1]) > self.size // 2:
                    gauss_bottom = self.size
                    prob_bottom = point[1] + self.size // 2 
                else:
                    gauss_bottom = self.height - point[1] + self.size // 2
                    prob_bottom = self.height

                if point[1] > self.size // 2:
                    gauss_top = 0
                    prob_top = point[1] - self.size // 2
                else:
                    gauss_top = self.size // 2 - point[1]
                    prob_top = 0
  
                prob[int(prob_left):int(prob_right),int(prob_top):int(prob_bottom)] = self.gaussian[int(gauss_left):int(gauss_right),int(gauss_top):int(gauss_bottom)]
                prob_sum += prob

            prob_sum = (prob_sum / np.max(prob_sum)).T
            #If prob_sum < 0.5, set it to 0
            prob_sum[prob_sum < 0.25] = 0
            print(prob_sum.shape, 'aqui pasa algo')
            #prob_sum[prob_sum > 0.25] = 1
            prob_paint = np.expand_dims((prob_sum), axis=2)
            print('uy la prob paint', prob_paint.shape)
            color = np.array(self.colormap_interactions[cluster]).reshape(1, 3)
            print(prob_paint.shape)
            prob_paint = (prob_paint @ color).astype(np.uint8)
            print(prob_paint.shape)
            print(img_copy.shape)
            print(color.shape)
            img_copy = cv2.addWeighted(img_copy, 0.5, prob_paint, 2.0, 0)
            cv2.imwrite(os.path.join(self.output_clusters, img_name + ' ' + cluster + '.png'), img_copy)
            print('Saved image', os.path.join(self.output_clusters, img_name + ' ' + cluster + '.png'))
            img_copy = cv2.imread(os.path.join(self.imgs_dir, img_name + '.jpg'))
        #Draw the text of the clusters for a better visualization
        for i in range(len(self.interaction_clusters)):
            cluster = self.interaction_clusters[i]
            for j in range(len(self.interaction_coordinates[cluster])):
                if j == 0:
                    point = self.interaction_coordinates[cluster][j]
                    cv2.putText(img_copy, cluster, (int(point[0]), int(point[1])), font, 3, (255,0,0), 3, cv2.LINE_AA)    
        cv2.imwrite(os.path.join(self.output_clusters, img_name + '------' + '.png'), img_copy)   

    def get_gaussian(self):
        x, y = np.mgrid[0:self.size:1, 0:self.size:1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        gaussian = multivariate_normal(mean=[self.size//2, self.size//2], cov=np.eye(2)*1000)
        return gaussian.pdf(pos)

    def examine_sequence(self):
        c = 0
        for root, dirs, files in os.walk(self.labels_dir):
            for file in files:
                if file.endswith('.pkl'):
                    pkl = open(os.path.join(root, file), 'rb')
                    print('Processing file', os.path.join(root, file))
                    data = pickle.load(pkl)

                    img = cv2.imread(os.path.join(self.imgs_dir, data['filename']))
                    
                    t_c, R_c = self.get_camera_pose(data)
                    points_in_camera = R_c @ self.points_coord.T + t_c
                    objects_in_scene = self.object_detector_gt(data['VISOR'])
                    
                    reprojected_points = self.reproject_points(points_in_camera)
                    self.filter_reprojected_points(reprojected_points, objects_in_scene, data['filename'])
                    #self.paint_points(img, data['filename'])
                    self.cluster_interactions()
                    self.paint_clusters(img, data['filename'])
                    pkl.close()
                    break
                    c += 1
                if c % 100 == 0:
                    print(c, 'images processed')




reproject = Reproject_data()
the_3D_points = reproject.examine_sequence()
