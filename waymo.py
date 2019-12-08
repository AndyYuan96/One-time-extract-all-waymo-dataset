import os
import tensorflow as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def yzl_convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0):
  """Convert range images to point cloud.

  Args:
    frame: open dataset frame
     range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
       camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    points: {[N, 12]} list of 3d lidar points of length 5 (number of lidars).
    x y z intensity elongation is_in_nlz
    channel 6: camera name
    channel 7: x (axis along image width)
    channel 8: y (axis along image height)
    channel 9: camera name of 2nd projection (set to UNKNOWN if no projection)
    channel 10: x (axis along image width)
    channel 11: y (axis along image height)

  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []
  cp_points = []

  frame_pose = tf.convert_to_tensor(
      value=np.reshape(np.array(frame.pose.transform), [4, 4]))
  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(value=range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == open_dataset.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_mask = range_image_tensor[..., 0] > 0 #tf.ones_like(range_image_tensor[..., 0])
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    range_image_cartesian = tf.concat([range_image_cartesian,range_image_tensor[..., 1:]],axis=2)

    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][0]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data,dtype=tf.float32), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor,
                                    tf.compat.v1.where(range_image_mask))

    points_tensor = tf.concat([points_tensor, cp_points_tensor],axis=1)

    points.append(points_tensor.numpy())

  return points

class WaymoSegment:
    def __init__(self,segment_path,root_path,train=1,debug=0):
        self.dataset=tf.data.TFRecordDataset(segment_path, compression_type='')
        if(train):
            self.base_path = root_path + "/" + "train/" + segment_path.split("/")[-1].split("_with")[0]
            self.all_label_path = "train/" + segment_path.split("/")[-1].split("_with")[0] + "/PCD_TOP/range1/"
        else:
            self.base_path = root_path + "/" + "val/" + segment_path.split("/")[-1].split("_with")[0]
            self.all_label_path = "val/" + segment_path.split("/")[-1].split("_with")[0] + "/PCD_TOP/range1/"

        self.debug = debug
        self.time_stamp = None
        self.save_name = None

        self._parse_frame()
        self._gen_path()

    def _parse_frame(self):
        self.frames = list()
        for data in self.dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.frames.append(frame)
            if(self.debug):
                break

    def _gen_path(self):
        self.pcd_path = ["PCD_TOP","PCD_FRONT","PCD_SIDE_LEFT","PCD_SIDE_RIGHT","PCD_REAR"]
        self.img_path = ["IMG_FRONT","IMG_FRONT_LEFT","IMG_FRONT_RIGHT","IMG_SIDE_LEFT","IMG_SIDE_RIGHT"]
        self.pcd_label_path = "LABEL"
        self.img_label_path = ["IMG_FRONT_LABEL","IMG_FRONT_LEFT_LABEL","IMG_FRONT_RIGHT_LABEL","IMG_SIDE_LEFT_LABEL","IMG_SIDE_RIGHT"]
        self.pose_path = "POSES"
        self.calib_path = "CALIB"

        for i in range(5):
            self.pcd_path[i] = self.base_path + "/" + self.pcd_path[i] + "/"
            self.img_path[i] = self.base_path + "/" + self.img_path[i] + "/"
            self.img_label_path[i] = self.base_path + "/" + self.img_label_path[i] +"/"
            if not os.path.exists(self.pcd_path[i]):
                os.makedirs(self.pcd_path[i]+"range1/")
                os.makedirs(self.pcd_path[i]+"range2/")

            if not os.path.exists(self.img_path[i]):
                os.makedirs(self.img_path[i])
            if not os.path.exists(self.img_label_path[i]):
                os.makedirs(self.img_label_path[i])

        self.pcd_label_path = self.base_path + "/" + self.pcd_label_path + "/"
        if not os.path.exists(self.pcd_label_path):
                os.makedirs(self.pcd_label_path)

        self.pose_path = self.base_path + "/" + self.pose_path + "/"
        if not os.path.exists(self.pose_path):
                os.makedirs(self.pose_path)

        self.calib_path = self.base_path + "/" + self.calib_path + "/"
        if not os.path.exists(self.calib_path):
                os.makedirs(self.calib_path)

        self.name_prefix = "pcd_"

    def save_all_frame(self,f):
        pbar = tqdm(total=len(self.frames))
        for frame in self.frames:
            self.save_one_frame(frame,f)
            pbar.update(1)
        pbar.close()

    def save_one_frame(self,frame,f):
        self.time_stamp = frame.timestamp_micros
        self.save_name = self.name_prefix + str(self.time_stamp)
        f.writelines(self.all_label_path + self.save_name + ".bin \n")
        self.save_point_cloud(frame)
        self.save_point_cloud_label(frame,f)
        self.save_img(frame)
        self.save_img_label(frame)
        self.save_poses(frame)
        self.save_img_poses(frame)



    def save_point_cloud(self,frame):
        (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        points = yzl_convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

        points_ri2 = yzl_convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)

        for i in range(len(points)):
            range1_pts = points[i]
            range2_pts = points_ri2[i]
            range1_pts.tofile(self.pcd_path[i] + "range1/" +self.save_name+".bin")
            range2_pts.tofile(self.pcd_path[i] + "range2/" + self.save_name+".bin")


    def save_img(self,frame):
        for image in frame.images:
            img_save_path = self.img_path[image.name - 1] + self.save_name + ".png"
            tmp = time.time()
            save_tensor = tf.image.decode_jpeg(image.image).numpy()
            save_tensor = cv2.cvtColor(save_tensor, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_save_path, save_tensor)
            print(time.time() - tmp)


    def save_point_cloud_label(self,frame,all_label_f,digits=4):
        labels = frame.laser_labels
        f = open(self.pcd_label_path + self.save_name+".txt",'w')
        for label in labels:
            ret = str(label.type) + " " + label.id + " " + \
            str(round(label.box.center_x,digits)) + " " + str(round(label.box.center_y,digits)) + " " + str(round(label.box.center_z,digits)) + " " + \
            str(round(label.box.length,digits)) + " " + str(round(label.box.width,digits)) + " " + str(round(label.box.height,digits)) + " " + \
            str(round(label.box.heading,digits)) + " " + \
            str(round(label.metadata.speed_x,digits)) + " " + str(round(label.metadata.speed_y,digits)) + " " + \
            str(round(label.metadata.accel_x,digits)) + " " + str(round(label.metadata.accel_x,digits)) + "\n"
            f.writelines(ret)

            all_label_ret = str(label.type) + " " + label.id + " " + \
            str(round(label.box.width,digits)) + " " + str(round(label.box.length,digits)) + " " + str(round(label.box.height,digits)) + " " + \
            str(round(label.box.center_x,digits)) + " " + str(round(label.box.center_y,digits)) + " " + str(round(label.box.center_z,digits)) + " " + \
            str(round(label.box.heading,digits)) + " " + \
            str(round(label.metadata.speed_x,digits)) + " " + str(round(label.metadata.speed_y,digits)) + " " + \
            str(round(label.metadata.accel_x,digits)) + " " + str(round(label.metadata.accel_x,digits)) + "\n"
            all_label_f.writelines(all_label_ret)

        f.close()

    def save_img_label(self,frame):
        for camera_labels in frame.camera_labels:
            camera_label_save_path = self.img_label_path[camera_labels.name - 1] + self.save_name + '.txt'
            f = open(camera_label_save_path,'w')
            for label in camera_labels.labels:
                ret = str(label.type) + " " + label.id + " " + \
                    str(label.box.center_x) + " " + str(label.box.center_y) + " " + \
                    str(label.box.width) + " " + str(label.box.length) + " " +  \
                    str(label.detection_difficulty_level) + " " + \
                    str(label.tracking_difficulty_level) + " " + "\n"
                f.writelines(ret)
            f.close()

    def save_poses(self,frame,digits=4):
        poses = frame.pose.transform
        save_poses_path = self.pose_path + self.save_name + ".txt"
        f = open(save_poses_path,'w')
        ret = ""
        ret += self.save_name + ".pcd "
        for i in range(len(poses)):
            ret += str(round(poses[i],digits)) + " "
        f.writelines(ret)
        f.close()

    def save_img_poses(self,frame):
        camera_calib_save_path = self.calib_path + self.save_name+".txt"
        f = open(camera_calib_save_path, 'w')
        calibrations = sorted(frame.context.camera_calibrations, key=lambda c: c.name)
        for calib in calibrations:
            f_u, f_v, c_u, c_v, k_1, k_2, p_1, p_2, k_3 = calib.intrinsic
            K = np.array([
                 [f_u, 0,   c_u],
                 [0,   f_v, c_v],
                 [0,   0,     1]])
            extrinic = np.array(calib.extrinsic.transform).reshape(4,4)
            extrinic = np.linalg.inv(extrinic)
            mid_matrix = np.array([1,0,0,0,0,1,0,0,0,0,1,0]).reshape(3,4)
            trans_matrix = np.dot(np.dot(K,mid_matrix),extrinic)
            ret = ""
            for i in range(3):
                for j in range(4):
                    ret += str(trans_matrix[i,j]) + " "

            ret = ret[:-1] + "\n"
            f.writelines(ret)
        f.close()

class Waymo():
    def __init__(self,origin_data_path, save_root_path,debug=0):
        self.origin_data_path = origin_data_path
        self.save_root_path = save_root_path
        self.train_segment_path = list()
        self.val_segment_path = list()
        self.debug = debug
        self._gen_segment_path()

    def _gen_segment_path(self):
        data_dirs = os.listdir(self.origin_data_path)
        for data_dir in data_dirs:
            if("tar" in data_dir):
                continue
            if(data_dir.split('_')[0] == "training"):
                train_root = self.origin_data_path + "/" + data_dir
                train_dirs = os.listdir(train_root)
                for train_dir in train_dirs[1:]:
                    self.train_segment_path.append(train_root + "/" + train_dir)
            elif(data_dir.split('_')[0] == "validation"):
                val_root = self.origin_data_path + "/" + data_dir
                val_dirs = os.listdir(val_root)
                for val_dir in val_dirs[1:]:
                    self.val_segment_path.append(val_root + "/" + val_dir)
        print("traing_segment_nums: " + str(len(self.train_segment_path)))
        print("val_segment_nums: " + str(len(self.val_segment_path)))

    def convert_all_segment(self):
        f = open(self.save_root_path+"/train.txt",'w')
        if(self.debug):
            self.train_segment_path = self.train_segment_path[:1]
        for train_segment in self.train_segment_path:
            print("parsing " + train_segment)
            ws = WaymoSegment(train_segment, self.save_root_path, debug=self.debug)
            ws.save_all_frame(f)
        f.close()

        f = open(self.save_root_path+"/val.txt",'w')
        if(self.debug):
            self.val_segment_path = self.val_segment_path[:1]
        for val_segment in self.val_segment_path:
            print("parsing " + val_segment)
            ws = WaymoSegment(val_segment, self.save_root_path, train=0, debug=self.debug)
            ws.save_all_frame(f)
        f.close()

if __name__ == "__main__":
    root_path = sys.argv[1]
    save_path = sys.argv[2]
    debug = sys.argv[3]
    debug = int(debug)
    wo = Waymo(root_path,save_path,debug)
    wo.convert_all_segment()


