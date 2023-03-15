import os
import numpy as np
import torch
import data_egom
import glob
import matplotlib.pyplot as plt
import open3d as o3d
import pickle
import cv2
import time


class Inference:
    def __init__(self):
        self.height = 480
        self.width = 854
        self.frame_idxs = [0]
        self.data_path = '...'
        self.kitchen = 'P03_EPIC_100'
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dataset = data_egom.VideoSequentialDataset(self.data_path, self.kitchen, self.height, self.width, self.frame_idxs)
        self.palette = self.dataset.colors
        self.output_dir = os.path.join(self.data_path, self.kitchen, '3D_output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.output_dir_2d = os.path.join(self.data_path, self.kitchen, 'aff_on_2d')
        if not os.path.exists(self.output_dir_2d):
            os.mkdir(self.output_dir_2d)

        self.alpha = 0.6
        self.depth_model_type = "DPT_Hybrid" #"DPT_Large"
        self.depth_model = torch.hub.load("intel-isl/MiDaS", self.depth_model_type)
        self.depth_model.to(self.device)
        self.depth_model.eval()
        self.depth_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        

    def depth_extractor(self, img, filename):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.depth_transforms(img).to(self.device)
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,).squeeze()
            disparity = prediction.cpu().numpy() 
            depth = 1 / disparity
        return depth

    def paint_affordance_hotpots(self, points, label): #SIMPLIFY THIS FUNCTION
        mask = np.zeros((self.height, self.width))
        aff = label
        aff_center = np.array(aff['affordance_center'])
        cv2.circle(mask, (int(aff_center[0]), int(aff_center[1])), 0, 1, -1)
        mask = mask.astype(bool)
        points = points[mask]
        
        painting_color = np.array(self.palette[label['verb_id']])
        img = (np.ones((self.height, self.width, 3)) * painting_color)[mask]
        return points, img 

    def obtain_rgbd(self, depth, scale):
        z = depth * scale
        x = (np.tile(np.arange(self.width), (self.height, 1)) - self.dataset.cx) * z / self.dataset.fx
        y = (np.tile(np.arange(self.height), (self.width, 1)).T - self.dataset.cy) * z / self.dataset.fy
        points = np.stack([x, y, z], axis=2) #h, w, 3
        return points

    def new_scale_SfM_depth(self, depth, colmap_depths, colmap_coords):
        SfM_depth, NN_depth = [], []
        for kypt in range(len(colmap_coords)):
            SfM_depth.append(colmap_depths[kypt]) #Interpretation 1 of the depth: La distancia entre el plano de la camara y el plano paralelo que corta el punto en 3D
            # Change order in coords, from XY to YX!!!
            u_interp = colmap_coords[kypt, 1] % 1
            v_interp = colmap_coords[kypt, 0] % 1
            u = int(colmap_coords[kypt, 1])
            v = int(colmap_coords[kypt, 0])
            if u < self.width - 1 and v < self.height - 1:
                interpolated_NN_depth = (1 - u_interp) * (1 - v_interp) * depth[v, u] + u_interp * (1 - v_interp) * depth[v, u + 1] + (1 - u_interp) * v_interp * depth[v + 1, u] + u_interp * v_interp * depth[v + 1, u + 1]
                NN_depth.append(interpolated_NN_depth)
            if u > self.width:
                print('alerta 1 !!!', u)
            if v > self.height:
                print('alerta  2 !!!', v)
        local_scale = np.median(np.array(SfM_depth)) / np.median(np.array(NN_depth))
        return local_scale
    
    def image_to_extrinsics(self, img):
        Rc, tc = img.qvec2rotmat(), img.tvec
        t = -Rc.T @ tc
        R = Rc.T
        return R, t
    
    def paint_aff_on_2D(self, img, frame_dict, keypoints):
        label = frame_dict["aff_annotation"]
        img_name = frame_dict['filename']
        ep100_label = frame_dict['EP100_annotation']
        visors_objects = frame_dict['VISOR_annotation']
        font = cv2.FONT_HERSHEY_PLAIN
        c = 0 
        if label is not None:   
            for h in range(len(label['hands'])):
                hand_bbox = label['hands'][h]['hand_bbox']
                cv2.rectangle(img, (hand_bbox[0], hand_bbox[1]), (hand_bbox[2], hand_bbox[3]), color=(0, 255, 0), thickness=2)
            for o in range(len(label['neutral_objects'])):
                obj_bbox = label['neutral_objects'][o]['noun_bbox']
                cv2.rectangle(img, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), color=(0, 255, 255), thickness=2)
            for aff_o in range(len(label['interacting_objects'])):
                obj_bbox = label['interacting_objects'][aff_o]['noun_bbox']
                cv2.rectangle(img, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), color=(255, 0, 0), thickness=5)
                x_center, y_center = label['interacting_objects'][aff_o]['affordance_center']
                cv2.circle(img, (int(x_center), int(y_center)), radius=10, color=(255, 0, 0), thickness=15)
                text = 'The ' + label['interacting_objects'][aff_o]['hand'] + ' is ' + label['interacting_objects'][aff_o]['verb'] + ' the ' + label['interacting_objects'][aff_o]['noun']
                cv2.putText(img, text, (10, 30 * (c + 1)), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
                c += 1
        if keypoints is not None:
            for kp in range(len(keypoints)):
                cv2.circle(img, (int(keypoints[kp, 1]), int(keypoints[kp, 0])), radius=1, color=(255, 255, 255), thickness=1)
        if ep100_label is not None:
            for i in range(len(ep100_label)):
                text = 'EP100 origi is: ' + ep100_label[i]['non_remapped_noun'] + ' remapped: ' + ep100_label[i]['noun'] + ' The verb ' + ep100_label[i]['verb']
                cv2.putText(img, text, (10, 100 + i*20), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
        for i in range(len(visors_objects)):
            cv2.putText(img, 'Object given by VISOR ' + visors_objects[i], (10, 110 + 30 * (i + 1)), font, 1.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.output_dir_2d, img_name), img)  


    def run(self):
        all_abs_depth, all_abs_colors, cameras, all_keypoints, all_rgb_keypoints = [], [], [], [], []
        global_counter = 0
        for i in range(len(self.dataset)):
            output_all = {}
            frame_dict = self.dataset[i]

            print('---------We are analying the frame', i, '--------- corresponding to the image', frame_dict['filename'], '---------')
            try:
                v = next(v for v in self.dataset.imgs_Colmap.values() if v.name == frame_dict['filename'])
            except:
                if frame_dict['exists_affordance']:
                    global_counter += 1
                print('We lost the camera pose for this image')
                continue

            R, t = self.image_to_extrinsics(v) #Location of camera with respect to the world
            cameras.append(t)
            label_t = frame_dict["aff_annotation"]
            colmap_coords = None
            output_all['EGOMETRIC_label'] = {'affordance_labels': []}
            if frame_dict['exists_affordance']:
                colmap_depths = np.array([(v.qvec2rotmat() @ self.dataset.pts_Colmap[p3d].xyz + v.tvec)[2] for p3d in v.point3D_ids[v.point3D_ids > -1]]) #WE PASS TO CAMERA COORDINATES
                colmap_coords = np.array([v.xys[np.where(v.point3D_ids == p3d)][0, ::-1] for p3d in v.point3D_ids[v.point3D_ids > -1]]) #Depth of the keypoints in the camera coordinates
                colmap_keypoints = np.array([self.dataset.pts_Colmap[p3d].xyz for p3d in v.point3D_ids[v.point3D_ids > -1]]) #Absolute coordinates
                colmap_rgb = np.array([self.dataset.pts_Colmap[p3d].rgb for p3d in v.point3D_ids[v.point3D_ids > -1]]) #Absolute coordinates
                colmap_rgb = self.alpha * colmap_rgb + (1 - self.alpha) * 255
                all_keypoints.append(colmap_keypoints)
                all_rgb_keypoints.append(colmap_rgb)
                depth = self.depth_extractor(frame_dict[('color', 0)], frame_dict['filename']) #Depth map in image coordinates (Relative!!)
                local_scale = self.new_scale_SfM_depth(depth, colmap_depths, colmap_coords)
                rescaled_rgbd = self.obtain_rgbd(depth, local_scale)

                for aff in range(len(label_t['interacting_objects'])):
                    rel_points, rel_colors = self.paint_affordance_hotpots(rescaled_rgbd, label_t['interacting_objects'][aff])
                    abs_points = np.dot(R, rel_points.reshape(-1, 3).T).T + t
                    abs_colors = np.reshape(rel_colors, (-1, 3))
                    #abs_points = np.concatenate((abs_points, abs_points + np.random.randn(20, 3) * 0.1), axis=0)
                    #abs_colors = np.concatenate((abs_colors, abs_colors + np.random.randn(20, 3) * 0), axis=0)
                    all_abs_depth.append(abs_points) 
                    all_abs_colors.append(abs_colors)
                    dict_aff = {'3D_aff_points': abs_points, 
                                '3D_aff_colors': abs_colors, 
                                'aff_noun': label_t['interacting_objects'][aff]['noun'], 
                                'aff_noun_id': label_t['interacting_objects'][aff]['noun_id'], 
                                'aff_verb': label_t['interacting_objects'][aff]['verb'], 
                                'aff_verb_id': label_t['interacting_objects'][aff]['verb_id']}
                    output_all['EGOMETRIC_label']['affordance_labels'].append(dict_aff)
                    #output_all['EGOMETRIC_label'][aff]['aff_' + str(aff)] = abs_points
                    #output_all['EGOMETRIC_label'][aff]['aff_rgb_' + str(aff)] = abs_colors
                    #output_all['EGOMETRIC_label'][aff]['aff_noun_' + str(aff)] = label_t['interacting_objects'][aff]['noun']
                    #output_all['EGOMETRIC_label'][aff]['aff_noun_id_' + str(aff)] = label_t['interacting_objects'][aff]['noun_id']
                    #output_all['EGOMETRIC_label'][aff]['aff_verb_' + str(aff)] = label_t['interacting_objects'][aff]['verb']
                    #output_all['EGOMETRIC_label'][aff]['aff_verb_id_' + str(aff)] = label_t['interacting_objects'][aff]['verb_id']
            #self.paint_aff_on_2D(frame_dict[('color', 0)], frame_dict, colmap_coords)
            output_all['colmap'] = {}  
            output_all['colmap']['keypoints_3D'] = colmap_keypoints
            output_all['colmap']['keypoints_rgb'] = colmap_rgb
            output_all['colmap']['keypoints_2D'] = colmap_coords
            output_all['colmap']['R_pos'] = R #Rotation matrix of the camera
            output_all['colmap']['t_pos'] = t #Translation vector of the camera
            output_all['VISOR'] = {}
            if label_t is not None:
                output_all['VISOR']['neutral_objects'] = label_t['neutral_objects']
                output_all['VISOR']['hands'] = label_t['hands']
                output_all['VISOR']['interacting_objects'] = label_t['interacting_objects']
                output_all['EPIC_100'] = frame_dict['EP100_annotation']
            output_all['filename'] = frame_dict['filename'] #Name of the image
            output_all['sequence'] = frame_dict['sequence'] #Name of the sequence

            #Save the output_all in a json file
            output_filename = os.path.join(self.output_dir, 'affordances_' + output_all['filename'].split('.')[0] +'.pkl')
            #With pickle
            with open(output_filename, 'wb') as f:
                pickle.dump(output_all, f)
            

        #Plot the camera pose and the sparse point cloud with Matplotlib
        cameras = np.array(cameras)
        keypoints = np.concatenate(all_keypoints, axis=0)
        abs_depth = np.concatenate(all_abs_depth, axis=0)
        abs_colors = np.concatenate(all_abs_colors, axis=0)
        rgb_keypoints = np.concatenate(all_rgb_keypoints, axis=0)
        print(keypoints.shape, abs_depth.shape, abs_colors.shape, rgb_keypoints.shape)
        
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='r')
        #ax.scatter(cameras[:, 0], cameras[:, 1], cameras[:, 2], c='b')
        #ax.scatter(origin[0], origin[1], origin[2], c='k')
        #plt.show()
                
        #Plot the camera pose and the sparse point cloud with Open3D
        pcd_plot = o3d.geometry.PointCloud()
        #Draw these points bigger
        pcd_plot.points = o3d.utility.Vector3dVector(abs_depth)
        pcd_plot.colors = o3d.utility.Vector3dVector(abs_colors / 255.0)
        cameras_plot = o3d.geometry.PointCloud()
        cameras_plot.points = o3d.utility.Vector3dVector(cameras)
        cameras_plot.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1] for i in range(cameras.shape[0])]))
        keypoints_plot = o3d.geometry.PointCloud()
        keypoints_plot.points = o3d.utility.Vector3dVector(keypoints)
        keypoints_plot.colors = o3d.utility.Vector3dVector(rgb_keypoints / 255.0)
        o3d.visualization.draw_geometries([cameras_plot, pcd_plot, keypoints_plot], height = 800, width = 1200)
   
        
        
inf = Inference()
data = inf.run()
                                

    
    
