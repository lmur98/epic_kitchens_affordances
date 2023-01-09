import os
import cv2
import numpy as np
import ijson
import json
import matplotlib.pyplot as plt
import pandas as pd
import time


class EP100_and_VISOR_annotations():
    def __init__(self, VISOR_path, img_dir, kitchen):
        #Visor dataset
        self.VISOR_json_dir_dense = os.path.join(VISOR_path, 'Interpolations-DenseAnnotations', 'train')
        self.VISOR_json_dir_sparse = os.path.join(VISOR_path, 'GroundTruth-SparseAnnotations', 'annotations', 'train')
        self.kitchen = kitchen.split('_')[0]

        self.all_dense_VISOR_jsons = {}
        self.all_sparse_VISOR_jsons = {}
        for root, dirs, files in os.walk(self.VISOR_json_dir_dense):
            for file in files:
                kitchen_name = file.split('_')[0]
                sequence_name = file.split('_')[1]
                if kitchen_name == self.kitchen and file.endswith('.json'):
                    dense_file = os.path.join(self.VISOR_json_dir_dense, file)
                    sparse_file = os.path.join(self.VISOR_json_dir_sparse, file)[:-20] + '.json'
                    self.all_sparse_VISOR_jsons[kitchen_name + '_' + sequence_name] =  sparse_file
                    self.all_dense_VISOR_jsons[kitchen_name + '_' + sequence_name] =  dense_file
        print(self.all_sparse_VISOR_jsons)
        self.hands = ['left hand', 'hand:left', 'right hand', 'hand:right']
        
        #Read the EPIC-Kitchen 100 narration
        self.EPIC_100_pkl = os.path.join(VISOR_path, 'EPIC_100_train.pkl')
        self.EPIC_100_narration = pd.read_pickle(self.EPIC_100_pkl)

        #Dictionary to remap the VISOR and EPIC-100 classes
        self.EPIC_100_nouns = os.path.join(VISOR_path, 'EPIC_100_noun_classes_v2.csv')
        self.EPIC_100_nouns = pd.read_csv(self.EPIC_100_nouns)

        #Directory with the sampled images, which we have their colmap poses
        self.img_dir = img_dir
        
    def affordance_hotspot(self, img_name, subset, sequence):
        #Output dictionary with the bounding boxes of the interacting hands and objects
        output = {'neutral_objects': [], 'interacting_objects': [], 'hands': []}
        VISOR_active_objects_list = []
        frame_id = int(img_name.split('.')[0].split('_')[-1])
        EP100_narration_list = self.read_EPIC_100_annot(frame_id, sequence)
        if EP100_narration_list is not None: #If there is a narration for the frame
            print('que hacen leer', sequence, img_name)
            VISOR_active_objects, divisor = self.read_VISOR_annot(img_name, subset, sequence) #Read the VISOR annotations
            if VISOR_active_objects is not None: #If there is a VISOR annotation for the frame
                for narration in range(len(EP100_narration_list)): #We can have multiple narrations for the same frame
                    EP100_narration = EP100_narration_list[narration] #Read the EPIC-100 narration
                    for e_idx, entity in enumerate(VISOR_active_objects): #Read the VISOR annotations
                        VISOR_active_objects_list.append(entity['name']) #To show later the active objects in the image
                        if entity['name'] in self.hands:
                            hand_bbox = self.get_bbox_from_segment(entity['segments']) #Add the bounding box of the hand
                            output['hands'].append({'hand': entity['name'], 'hand_bbox': tuple([int(item / divisor) for item in hand_bbox])})
                            for e_idx2, entity_2 in enumerate(VISOR_active_objects): #VISOR annotations are 'name', but when we remapp them we call 'noun', as well as with EP100
                                entity_2_name = self.remap_VISOR_annot(entity_2)['noun']
                                if entity_2_name in self.hands:
                                    continue
                                elif entity_2_name in EP100_narration['noun']:
                                    obj_bbox = self.get_bbox_from_segment(entity_2['segments'])
                                    cond_aff_intersect, aff_bbox = self.get_intersection_bbox(hand_bbox, obj_bbox)
                                    if cond_aff_intersect:
                                        x_center, y_center = self.get_bbox_center(aff_bbox)
                                        output['interacting_objects'].append({'hand': entity['name'],
                                                                            'verb': EP100_narration['verb'],
                                                                            'verb_id': EP100_narration['verb_id'],
                                                                            'noun': entity_2_name, 
                                                                            'noun_id': EP100_narration['noun_id'],
                                                                            'noun_bbox': tuple([int(item / divisor) for item in obj_bbox]), 
                                                                            'hand_bbox': tuple([int(item / divisor) for item in hand_bbox]),
                                                                            'affordance_bbox': tuple([int(item / divisor) for item in aff_bbox]), 
                                                                            'affordance_center': (int(x_center / divisor), int(y_center / divisor))})
                                        print('There is an interaction!!!:))')
                                    else:
                                        output['neutral_objects'].append({'noun': entity_2_name, 'noun_bbox': tuple([int(item / divisor) for item in obj_bbox])})
                                else:
                                    output['neutral_objects'].append({'noun': entity_2_name, 'noun_bbox': tuple([int(item / divisor) for item in self.get_bbox_from_segment(entity_2['segments'])])})
            #Check that if there is not any interacting objects, we add the interaction in the center of the hand bounding box
            #if len(output['interacting_objects']) == 0:
            
        else:
            output = None
        return output, EP100_narration_list, VISOR_active_objects_list

    def affordance_hotspot_visual(self, img_name):
        #Output dictionary with the bounding boxes of the interacting hands and objects
        output = {'neutral_objects': [], 'interacting_objects': [], 'hands': []}
        self.img_path = os.path.join(self.img_dir, img_name + '.jpg')
        frame_id = int(img_name.split('_')[-1])
        VISOR_active_objects = self.read_VISOR_annot()
        EP100_narration_list = self.read_EPIC_100_annot(frame_id, sequence)
        self.img_show = cv2.imread(self.img_path)
        for narration in EP100_narration_list:
            EP100_narration = EP100_narration_list[narration]
            for e_idx, entity in enumerate(VISOR_active_objects):
                if entity['noun'] in self.hands:
                    hand_bbox = self.get_bbox_from_segment(entity['segments'])
                    output['hands'].append({'hand': entity['noun'], 'hand_bbox': hand_bbox})
                    cv2.rectangle(self.img_show, (hand_bbox[0], hand_bbox[1]), (hand_bbox[2], hand_bbox[3]), color=(0, 0, 255), thickness=10)
                    for e_idx2, entity_2 in enumerate(VISOR_active_objects):
                        entity_2_name = self.remap_VISOR_annot(entity_2)['noun']
                        if entity_2_name in self.hands:
                            continue
                        elif entity_2_name in EP100_narration['noun']:
                            obj_bbox = self.get_bbox_from_segment(entity_2['segments'])
                            cv2.rectangle(self.img_show, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), color=(0, 255, 0), thickness=10)
                            if self.get_intersection_bbox(hand_bbox, obj_bbox)[0]:
                                aff_bbox = self.get_intersection_bbox(hand_bbox, obj_bbox)[1]
                                x_center, y_center = self.get_bbox_center(aff_bbox)
                                output['interacting_objects'].append({'hand': entity['noun'],'verb': EP100_narration['verb'],'object': entity_2_name, 'object_bbox': obj_bbox, 'hand_bbox': hand_bbox,'affordance_bbox': aff_bbox, 'affordance_center': (x_center, y_center)})
                                cv2.circle(self.img_show, (int(x_center), int(y_center)), radius=10, color=(255, 0, 0), thickness=15)
                            else:
                                output['neutral_objects'].append({'object': entity_2_name, 'object_bbox': obj_bbox})
                        else:
                            obj_bbox = self.get_bbox_from_segment(entity_2['segments'])
                            output['neutral_objects'].append({'object': entity_2_name, 'object_bbox': obj_bbox})
                            cv2.rectangle(self.img_show, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), color=(0, 255, 255), thickness=2)
            cv2.imwrite('/home/lmur/Documents/Monodepth/ego_metric/affordance_hotspot3.jpg', self.img_show)
        sampled_mask = os.path.join('/home/lmur/Desktop/EGO_METRIC_DATASET_V2/dense_masks', self.sequence, img_name + '.png')
        sampled_mask = cv2.imread(sampled_mask)
        cv2.imwrite('/home/lmur/Documents/Monodepth/ego_metric/active_object_masks3.jpg', sampled_mask)    
        return output


    def read_VISOR_annot(self, img_name, subset, sequence):
        print('PROBAMOS CON EL SPARSE')
        VISOR_filename = self.all_sparse_VISOR_jsons[sequence]
        the_annotation = None
        with open(VISOR_filename, 'r') as f:
            VISOR_annot = ijson.items(f, 'video_annotations.item')
            for entity in VISOR_annot:
                if entity['image']['name'].split('.')[0] == img_name.split('.')[0]:
                    the_annotation = entity['annotations']
                    divisor = 2.25
                    break
        if the_annotation is None:
            print('PROBANMOS CON EL DENSE')
            VISOR_filename = self.all_dense_VISOR_jsons[sequence]
            with open(VISOR_filename, 'r') as f:
                VISOR_annot = ijson.items(f, 'video_annotations.item')
                for entity in VISOR_annot:
                    if entity['image']['name'].split('.')[0] == img_name.split('.')[0]:
                        the_annotation = entity['annotations']
                        divisor = 1
                        break
        return the_annotation, divisor

    def remap_VISOR_annot(self, visor_annot):
        visor_noun_class = visor_annot['class_id']
        remapped = self.EPIC_100_nouns[self.EPIC_100_nouns['id'] == visor_noun_class]
        full_visor_annot = {'noun_id': remapped['id'].values[0],
                            'noun': remapped['key'].values[0],
                            'category': remapped['category'].values[0],
                            'non_remapped_noun': visor_annot['name']}
        return full_visor_annot

    def read_EPIC_100_annot(self, frame_id, sequence):
        df = self.EPIC_100_narration
        df = df[df['video_id'] == sequence]
        df = df.reset_index(drop=True)
        EP_100_narration = df[(df['start_frame'] <= frame_id) & (df['stop_frame'] >= frame_id)]
        if len(EP_100_narration) == 0:
            return None
        list_annotations = []
        for i in range(len(EP_100_narration)):
            EP_100_narration_noun = EP_100_narration['noun_class'].values[i]
            remapped = self.EPIC_100_nouns[self.EPIC_100_nouns['id'] == EP_100_narration_noun]
            narration_annot = {'noun_id': remapped['id'].values[0], 
                               'noun': remapped['key'].values[0],
                               'category': remapped['category'].values[0],
                               'non_remapped_noun': EP_100_narration['noun'].values[i],
                               'verb': EP_100_narration['verb'].values[i],
                               'verb_id': EP_100_narration['verb_class'].values[i]}
            list_annotations.append(narration_annot)
        return list_annotations

    def get_bbox_from_segment(self, annot):
        mask_clean = []
        for mask in annot:
            if len(mask) == 0: continue
            mask = np.array(mask, dtype=np.int32)
            mask_clean.append(mask)
        bbox = self.get_bbox(mask_clean)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        return x1, y1, x2, y2

    def get_bbox(self, masks):
        '''
        Get bbox for object masks (1 object may have 1> components). Returns:
        bbox: [x, y, height, width]
        '''
        g_xmin, g_ymin, g_xmax, g_ymax = 10000, 10000, 0, 0
        for mask in masks:
            if len(mask) == 0: continue
            mask = np.array(mask)
            xmin, xmax = np.min(mask[:,0]), np.max(mask[:,0])
            ymin, ymax = np.min(mask[:,1]), np.max(mask[:,1])

            g_xmin = min(g_xmin, xmin)
            g_xmax = max(g_xmax, xmax)
            g_ymin = min(g_ymin, ymin)
            g_ymax = max(g_ymax, ymax)

        bbox = [int(g_xmin), int(g_ymin), int(g_xmax - g_xmin), int(g_ymax - g_ymin)]
        return bbox

    def get_intersection_bbox(self, hand_bbox, obj_bbox):
        x1, y1, x2, y2 = hand_bbox
        x3, y3, x4, y4 = obj_bbox
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        intersection_bbox = [x_left, y_top, x_right, y_bottom]
        if x_right < x_left or y_bottom < y_top:
            return False, None
        else:
            return True, intersection_bbox

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox #Get the center of the affordance hotspot
        x_center = x1 + (x2 - x1)/2
        y_center = y1 + (y2 - y1)/2
        return x_center, y_center
