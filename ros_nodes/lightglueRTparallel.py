#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import argparse
import sys
import time
import yaml

from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

from lightglue_dynamo.preprocessors import SuperPointPreprocessor

import rospy
import message_filters as mf
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2 as PointCloud2Msg
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2


# Project points in 3D
def unproject(image, kpt_left, kpt_right, camera_params): 
    
    fx1, fy1 = camera_params['fx1'], camera_params['fy1']
    cx1, cy1 = camera_params['cx1'], camera_params['cy1']
    cx2, baseline = camera_params['cx2'], camera_params['baseline']

    # Transform disparity in meters
    disparity = kpt_left[:, 0] - kpt_right[:, 0]
    depth = (fx1 * baseline) / (disparity + (cx2 - cx1))

    # Remove negative depths
    valid_depth_mask = depth > 0
    kpt_left = kpt_left[valid_depth_mask]
    depth = depth[valid_depth_mask]
    
    # Projection into 3D
    x = (kpt_left[:, 0] - cx1) * depth / fx1
    y = (kpt_left[:, 1] - cy1) * depth / fy1
    z = depth    
    rgb = image[kpt_left[:, 1], kpt_left[:, 0]]

    # Pack RGB into uint32 using NumPy
    a = np.full((rgb.shape[0],), 255, dtype=np.uint8)
    r, g, b = rgb[:, 2], rgb[:, 1], rgb[:, 0]  # BGR to RGB
    rgba = (a.astype(np.uint32) << 24) | (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    
    # Combine all fields into final Nx4 array
    points = list(zip(x.tolist(), y.tolist(), z.tolist(), rgba.tolist()))

    pointcloud_msg = PointCloud2Msg()
    pointcloud_msg.header.stamp = rospy.Time.now()
    pointcloud_msg.header.frame_id = "camera_frame"

    pointcloud_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
    ]
    pointcloud_msg = point_cloud2.create_cloud(pointcloud_msg.header, pointcloud_msg.fields, points)
    return pointcloud_msg

def makeMatchesImages(im1, im2, kpts, matches, mscores):
    """
    Create a visualization image with keypoints and matches.
    """

    kpts1 = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in kpts[0]]
    kpts2 = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in kpts[1]]

    # Create matches and sort them by score
    matches1to2 = [cv2.DMatch(_queryIdx=int(m[1]), _trainIdx=int(m[2]), _imgIdx=0, _distance=float(ms)) for m, ms in zip(matches, mscores)]
    matches1to2 = sorted(matches1to2, key=lambda x: x.distance, reverse=True)
    printable_matches = min(len(matches1to2), 100)
    sparse_disp = cv2.drawMatches(img1=im1, keypoints1=kpts1,
                                  img2=im2, keypoints2=kpts2,
                                  matches1to2=matches1to2[:printable_matches], outImg=None,
                                  matchColor=(0, 255, 0), singlePointColor=None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return sparse_disp

def create_params_dict(camera_yaml, factor=4.0):
    camera_params = {}
    with open(camera_yaml, 'r') as f:
        stero_params = yaml.safe_load(f)

    inv_f = 1.0 / factor
    camera_params['fx1'] = float(stero_params['fx1']) * inv_f
    camera_params['fy1'] = float(stero_params['fy1']) * inv_f
    camera_params['cx1'] = float(stero_params['cx1']) * inv_f
    camera_params['cy1'] = float(stero_params['cy1']) * inv_f
    camera_params['cx2'] = float(stero_params['cx2']) * inv_f
    camera_params['baseline'] = float(stero_params['baseline'])

    return camera_params

def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = []
    if img_msg.encoding == "mono8":
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width), # and one channel of data.
                                    dtype=dtype, buffer=img_msg.data)
    if img_msg.encoding == "bgr8":
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                                dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image, encoding="passthrough"):
    img_msg = ImageMsg()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = encoding
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

class RosLightGlueWrapper:
  def __init__(self, args):
    self.args = args

    # Store stereo parameters
    self.left_stereo = create_params_dict(f'{args.camera_folder}/stereo_left_air.yaml', factor=2.0)
    self.right_stereo = create_params_dict(f'{args.camera_folder}/stereo_right_air.yaml', factor=2.0)
    
    build_engine = EngineFromBytes(BytesFromPath(str(args.path2engine)))
    self.model = TrtRunner(build_engine)
    self.model.__enter__()

    # Warm-up inference
    print("Running warm-up inference...")
    img_left = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    img_right = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    images = np.stack([img_left, img_right])
    images = SuperPointPreprocessor.preprocess(images)
    images = images.astype(np.float32)
    for _ in range(2):
        self.model.infer(feed_dict={"images": images})
    print("Warm-up inference done.")

    # IMAGE SUBSCRIBERS
    self.left_left_sub = mf.Subscriber(f'/stereo/stereo_left/left/image_rect', ImageMsg)
    self.left_right_sub = mf.Subscriber(f'/stereo/stereo_left/right/image_rect', ImageMsg)
    self.right_left_sub = mf.Subscriber(f'/stereo/stereo_right/left/image_rect', ImageMsg)
    self.right_right_sub = mf.Subscriber(f'/stereo/stereo_right/right/image_rect', ImageMsg) 
    
    # POINT CLOUD PUBLISHERS
    self.pcl_left_pub  = rospy.Publisher(f'/elas_left/point_cloud', PointCloud2Msg, queue_size=1)
    self.pcl_right_pub = rospy.Publisher(f'/elas_right/point_cloud', PointCloud2Msg, queue_size=1)

    self.ts = mf.ApproximateTimeSynchronizer([self.left_left_sub, self.left_right_sub, 
                                              self.right_left_sub, self.right_right_sub], queue_size=1, slop=0.1)
    self.ts.registerCallback(self.run_model)

    if args.debug:
        self.disp_left_pub = rospy.Publisher(f'/elas_left/disparity', ImageMsg, queue_size=1)
        self.disp_right_pub = rospy.Publisher(f'/elas_right/disparity', ImageMsg, queue_size=1)

    print("Stereo model initialized and ready to process images.")

  def run_model(self, left_left_img_msg: ImageMsg,  left_right_img_msg: ImageMsg,
                      right_left_img_msg: ImageMsg, right_right_img_msg: ImageMsg):
    
    # Convert ROS Image messages to OpenCV images
    left_left_img   = imgmsg_to_cv2(left_left_img_msg)
    left_right_img  = imgmsg_to_cv2(left_right_img_msg)
    right_left_img  = imgmsg_to_cv2(right_left_img_msg)
    right_right_img = imgmsg_to_cv2(right_right_img_msg)

    # Resize images to the model's input size
    if left_left_img.shape[-2] != self.args.height or left_left_img.shape[-1] != self.args.width: 
        left_left_img = cv2.resize(left_left_img, (self.args.width, self.args.height))
        left_right_img = cv2.resize(left_right_img, (self.args.width, self.args.height))
        right_left_img = cv2.resize(right_left_img, (self.args.width, self.args.height))
        right_right_img = cv2.resize(right_right_img, (self.args.width, self.args.height))

    # Preprocess images
    raw_images_left, raw_images_right  = [left_left_img, left_right_img], [right_left_img, right_right_img]
    images_left, images_right = np.stack(raw_images_left), np.stack(raw_images_right)
    images_left, images_right = SuperPointPreprocessor.preprocess(images_left), SuperPointPreprocessor.preprocess(images_right) 
    images_left, images_right = images_left.astype(np.float32), images_right.astype(np.float32)

    # Run inference
    time_start = time.time()

    outputs_left = self.model.infer(feed_dict={"images": images_left})
    kpts_left, matches_left, mscores_left = np.copy(outputs_left["keypoints"]), np.copy(outputs_left["matches"]), np.copy(outputs_left["mscores"])
    outputs_right = self.model.infer(feed_dict={"images": images_right})
    kpts_right, matches_right, mscores_right = np.copy(outputs_right["keypoints"]), np.copy(outputs_right["matches"]), np.copy(outputs_right["mscores"])
    
    total_inference_time = self.model.last_inference_time() * 2
    print(f"Inference Time: {total_inference_time:.3f} s")
    time_end = time.time()
    print(f"Total Time: {time_end - time_start:.3f} s")


    if self.args.debug:    
        sparse_disp_left  = makeMatchesImages(raw_images_left[0], raw_images_left[1], kpts_left, 
                                             matches_left, mscores_left)
        sparse_disp_right = makeMatchesImages(raw_images_right[0], raw_images_right[1], kpts_right, 
                                             matches_right, mscores_right)
        
        # opencv format
        sparse_disp_left_msg = cv2_to_imgmsg(sparse_disp_left, encoding="bgr8")
        sparse_disp_left_msg.header.stamp = left_left_img_msg.header.stamp
        sparse_disp_left_msg.header.frame_id = left_left_img_msg.header.frame_id

        sparse_disp_right_msg = cv2_to_imgmsg(sparse_disp_right, encoding="bgr8")
        sparse_disp_right_msg.header.stamp = right_left_img_msg.header.stamp
        sparse_disp_right_msg.header.frame_id = right_left_img_msg.header.frame_id
        
        # Publish sparse disparity image
        self.disp_left_pub.publish(sparse_disp_left_msg)
        self.disp_right_pub.publish(sparse_disp_right_msg)

        
    ## STEREO LEFT ##
    pcl_left_msg = unproject(raw_images_left[0], kpts_left[0][matches_left[..., 1]], 
                             kpts_left[1][matches_left[..., 2]], self.left_stereo)
    pcl_left_msg.header.stamp = left_left_img_msg.header.stamp
    pcl_left_msg.header.frame_id = left_left_img_msg.header.frame_id
    
    ## STEREO RIGHT ##
    pcl_right_msg = unproject(raw_images_right[0], kpts_right[0][matches_right[..., 1]], 
                              kpts_right[1][matches_right[..., 2]], self.right_stereo)
    pcl_right_msg.header.stamp = right_left_img_msg.header.stamp
    pcl_right_msg.header.frame_id = right_left_img_msg.header.frame_id
    
    self.pcl_left_pub.publish(pcl_left_msg)
    self.pcl_right_pub.publish(pcl_right_msg)

    del outputs_left, outputs_right, images_left, images_right
    
    return
    


def main():

    parser = argparse.ArgumentParser()
    
    # Parameters for the LightGlue model
    parser.add_argument('--path2engine', help="TensorRT Model", type=str, required=True)
    parser.add_argument('--camera_folder', help="Folder containing the camera params files", type=str, required=True)
    parser.add_argument('--width', type=int, default=640, help='Image width for the model')
    parser.add_argument('--height', type=int, default=400, help='Image height for the model')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args, _ = parser.parse_known_args()
    rospy.init_node(f'sparse_stereo_node', anonymous=True)
    
    try:
        lightglue_wrapper = RosLightGlueWrapper(args)
        rospy.spin()
    finally:
        lightglue_wrapper.model.__exit__(None, None, None)

    return

if __name__ == "__main__":
    main()
