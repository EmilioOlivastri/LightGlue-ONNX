#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import argparse
import sys
import struct
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
    with open(args.camera_params, 'r') as f:
        stereo_params = yaml.safe_load(f)
    self.fx1 = float(stereo_params['fx1']) * 0.5
    self.fy1 = float(stereo_params['fy1']) * 0.5
    self.cx1 = float(stereo_params['cx1']) * 0.5
    self.cy1 = float(stereo_params['cy1']) * 0.5
    self.cx2 = float(stereo_params['cx2']) * 0.5
    self.baseline = float(stereo_params['baseline'])

    build_engine = EngineFromBytes(BytesFromPath(str(args.path2engine)))
    self.model = TrtRunner(build_engine)
    # Warm-up inference
    print("Running warm-up inference...")
    img_left = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    img_right = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    images = np.stack([img_left, img_right])
    images = SuperPointPreprocessor.preprocess(images)
    images = images.astype(np.float32)
    with self.model:
        for _ in range(2):
            self.model.infer(feed_dict={"images": images})
    print("Warm-up inference done.")

    self.left_sub = mf.Subscriber(f'/stereo/{args.stereo_rig}/left/image_rect', ImageMsg)
    self.right_sub = mf.Subscriber(f'/stereo/{args.stereo_rig}/right/image_rect', ImageMsg) 
    
    side = args.stereo_rig.split('_')[1]
    self.pointcloud_pub = rospy.Publisher(f'/elas_{side}/point_cloud', PointCloud2Msg, queue_size=1)
    self.ts = mf.ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=1, slop=0.1)
    self.ts.registerCallback(self.run_model)

    if args.debug:
        self.sparse_disp_pub = rospy.Publisher(f'/stereo/{args.stereo_rig}/sparse_disparity', ImageMsg, queue_size=1)

    print("Stereo model initialized and ready to process images.")

  def run_model(self, left_img_msg: ImageMsg, right_img_msg: ImageMsg):
    # Convert ROS Image messages to OpenCV images
    left_image = imgmsg_to_cv2(left_img_msg)
    right_image = imgmsg_to_cv2(right_img_msg)

    # Resize images to the model's input size
    if left_image.shape[-2] != self.args.height or left_image.shape[-1] != self.args.width: 
        left_image = cv2.resize(left_image, (self.args.width, self.args.height))
        right_image = cv2.resize(right_image, (self.args.width, self.args.height))

    # Preprocess images
    raw_images = [left_image, right_image]
    images = np.stack(raw_images)
    images = SuperPointPreprocessor.preprocess(images)
    images = images.astype(np.float32)

    print(f"Left Image Shape: {left_image.shape}")

    # Run inference
    with self.model:
        outputs = self.model.infer(feed_dict={"images": images})
        print(f"Inference Time: {self.model.last_inference_time():.3f} s")

    keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]
    
    print(f'Matches: {matches[:, 1]}')

    if self.args.debug:    
        kpts_left = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints[0]]
        kpts_right = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints[1]]

        # Create matches and sort them by score
        matches1to2 = [cv2.DMatch(_queryIdx=int(m[1]), _trainIdx=int(m[2]), _imgIdx=0, _distance=float(ms)) for m, ms in zip(matches, mscores)]
        matches1to2 = sorted(matches1to2, key=lambda x: x.distance, reverse=True)
        printable_matches = min(len(matches1to2), 100)

        sparse_disp = cv2.drawMatches(img1=left_image, keypoints1=kpts_left,
                                      img2=right_image, keypoints2=kpts_right,
                                      matches1to2=matches1to2[:printable_matches], outImg=None,
                                      matchColor=(0, 255, 0), singlePointColor=None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # opencv format
        sparse_disp_msg = cv2_to_imgmsg(sparse_disp, encoding="bgr8")
        sparse_disp_msg.header.stamp = left_img_msg.header.stamp
        sparse_disp_msg.header.frame_id = left_img_msg.header.frame_id
        
        # Publish sparse disparity image
        self.sparse_disp_pub.publish(sparse_disp_msg)

        del kpts_left, kpts_right, matches1to2, sparse_disp

        
    pt_cloud_msg = self.unproject(raw_images[0], keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]])
    pt_cloud_msg.header.stamp = left_img_msg.header.stamp
    pt_cloud_msg.header.frame_id = left_img_msg.header.frame_id

    # Publish point cloud
    self.pointcloud_pub.publish(pt_cloud_msg)

    del outputs, images
    

  # Project points in 3D
  def unproject(self, image_left, kpt_left, kpt_right): 

    # Transform disparity in meters
    disparity = kpt_left[:, 0] - kpt_right[:, 0]
    depth = (self.fx1 * self.baseline) / (disparity + (self.cx2 - self.cx1))

    # Projection into 3D
    x = (kpt_left[:, 0] - self.cx1) * depth / self.fx1
    y = (kpt_left[:, 1] - self.cy1) * depth / self.fy1
    z = depth
    points3D = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = image_left[kpt_left[:, 1], kpt_left[:, 0]]
    
    # Create point cloud message
    pointcloud = []
    for i in range(points3D.shape[0]):
        a = 255
        b, g, r = colors[i][0], colors[i][1], colors[i][2]
        x, y, z = points3D[i][0], points3D[i][1], points3D[i][2]
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        pt = [x, y, z, rgb]
        pointcloud.append(pt)

    pointcloud_msg = PointCloud2Msg()
    pointcloud_msg.header.stamp = rospy.Time.now()
    pointcloud_msg.header.frame_id = "camera_frame"
    pointcloud_msg.height = 1
    pointcloud_msg.width = points3D.shape[0]
    
    pointcloud_msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
    ]
    pointcloud_msg = point_cloud2.create_cloud(pointcloud_msg.header, pointcloud_msg.fields, pointcloud)

    return pointcloud_msg


def main():

    parser = argparse.ArgumentParser()
    
    # Parameters for the LightGlue model
    parser.add_argument('--path2engine', help="TensorRT Model", type=str, required=True)
    parser.add_argument('--camera_params', help="camera parameters file", type=str, required=True)
    parser.add_argument('--width', type=int, default=640, help='number of flow-field updates during forward pass')
    parser.add_argument('--height', type=int, default=400, help='number of flow-field updates during forward pass')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    # Parameters for ROS node
    parser.add_argument('--stereo_rig', type=str, default='stereo_left', help='stereo rig name')
    print('Args:', sys.argv)

    args, unknown = parser.parse_known_args()
    rospy.init_node(f'sparse_{args.stereo_rig}_node', anonymous=True)
    lightglue_wrapper = RosLightGlueWrapper(args)
    rospy.spin()

    return

if __name__ == "__main__":
    main()
