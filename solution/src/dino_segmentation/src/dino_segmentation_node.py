#!/usr/bin/env python3

import yaml
import os

import numpy as np
import imgviz
import rospy
import rospkg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart
import cv2
from dino_segmentation.model import Wrapper
from cv_bridge import CvBridge
from integration import NUMBER_FRAMES_SKIPPED, filter_by_classes, filter_by_bboxes, filter_by_scores
from integration import get_steer_matrix_left_lane_markings, get_steer_matrix_right_lane_markings, detect_lane_markings

class ObjectDetectionNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        self.initialized = False
        self.log("Initializing!")

        self.veh = rospy.get_namespace().strip("/")
        self.avoid_duckies = False

        # Construct publishers
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic,
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )

        episode_start_topic = f"/{self.veh}/episode_start"
        rospy.Subscriber(episode_start_topic,
                         EpisodeStart,
                         self.cb_episode_start,
                         queue_size=1)

        self.pub_detections_image = rospy.Publisher(
            "~dino_segmentation_img", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Mask topics
        self.pub_left_mask = rospy.Publisher(
            "~lt_mask_seg", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        self.pub_right_mask = rospy.Publisher(
            "~rt_mask_seg", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )

        self.bridge = CvBridge()

        self.class2int = {'_background_': 0, 'yellow-lane': 1,
                          'white-lane': 2, 'duckiebot': 3,
                          'sign': 4, 'duck': 5, 'hand': 6}

        model_file = rospy.get_param('~model_file', '.')
        self.v = rospy.get_param('~speed', 0.4)
        self.veh = rospy.get_namespace().strip("/")
        aido_eval = rospy.get_param("~AIDO_eval", False)
        self.log(f"AIDO EVAL VAR: {aido_eval}")
        self.log("Starting model loading!")
        self._debug = rospy.get_param("~debug", False)
        self.model_wrapper = Wrapper(

        )
        self.log("Finished model loading!")
        self.frame_id = 0
        self.first_image_received = False
        self.initialized = True
        self.log("Initialized!")

    def cb_episode_start(self, msg: EpisodeStart):
        self.avoid_duckies = False
        self.pub_car_commands(True, msg.header)

    def image_cb(self, image_msg):
        if not self.initialized:
            self.pub_car_commands(True, image_msg.header)
            return

        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())
        if self.frame_id != 0:
            self.pub_car_commands(self.avoid_duckies, image_msg.header)
            return

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return

        old_img = image
        from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand
        if get_device_hardware_brand() != DeviceHardwareBrand.JETSON_NANO:  # if in sim
            if self._debug:
                print("Assumed an image was bgr and flipped it to rgb")
            old_img = image
            image = image[..., ::-1].copy()  # image is bgr, flip it to rgb

        old_img = cv2.resize(old_img, (480, 480))
        img = cv2.resize(image, (480, 480))
        pred_mask, class_names = self.model_wrapper.predict(image)

        # Resize the original image and the predictions to 480 x 480
        pred_mask = np.kron(pred_mask, np.ones((8, 8))).astype(int)  # Upscale the predictions back to 480x480

        # Create binary mask out of segmentation mask (TODO add weights per category)
        weighted_mask = np.zeros(pred_mask.shape)
        weighted_mask[:] = (pred_mask == self.class2int['white-lane']) + (pred_mask == self.class2int['yellow-lane']) + \
                           (pred_mask == self.class2int['duckiebot']) + (pred_mask == self.class2int['duck']) + \
                           (pred_mask == self.class2int['sign'])

        # Retrieve masks
        left_mask, right_mask = detect_lane_markings(weighted_mask)

        # detection = self.det2bool(bboxes, classes, scores)
        detection = True

        # as soon as we get one detection we will stop forever
        if detection:
            self.log("Duckie pedestrian detected... stopping")
            self.avoid_duckies = False  # True

        self.pub_car_commands(self.avoid_duckies, image_msg.header)

        if self._debug:
            viz = imgviz.label2rgb(
                pred_mask,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            # Publish predicted classes
            obj_det_img = self.bridge.cv2_to_imgmsg(viz, encoding="rgb8")
            self.pub_detections_image.publish(obj_det_img)
            # Publish left and right masks
            lt_mask_viz = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 0.2,
                                          left_mask.astype(np.uint8) * 255, 0.8, 0)
            rt_mask_viz = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 0.2,
                                          right_mask.astype(np.uint8) * 255, 0.8, 0)
            print('\n\n\n')
            print(np.unique(left_mask.astype(np.uint8)))
            print(np.unique(right_mask.astype(np.uint8)))
            print('\n\n\n')

            lt_mask_viz = self.bridge.cv2_to_imgmsg(lt_mask_viz, encoding="mono8")
            rt_mask_viz = self.bridge.cv2_to_imgmsg(rt_mask_viz, encoding="mono8")
            # Publish!
            self.pub_left_mask.publish(lt_mask_viz)
            self.pub_right_mask.publish(rt_mask_viz)

    def det2bool(self, bboxes, classes, scores):

        box_ids = np.array(list(map(filter_by_bboxes, bboxes))).nonzero()[0]
        cla_ids = np.array(list(map(filter_by_classes, classes))).nonzero()[0]
        sco_ids = np.array(list(map(filter_by_scores, scores))).nonzero()[0]

        box_cla_ids = set(list(box_ids)).intersection(set(list(cla_ids)))
        box_cla_sco_ids = set(list(sco_ids)).intersection(set(list(box_cla_ids)))

        if len(box_cla_sco_ids) > 0:
            return True
        else:
            return False

    def pub_car_commands(self, stop, header):
        car_control_msg = Twist2DStamped()
        car_control_msg.header = header
        if stop:
            car_control_msg.v = 0.0
        else:
            car_control_msg.v = self.v

        # always drive straight
        car_control_msg.omega = 0.0

        self.pub_car_cmd.publish(car_control_msg)

    def read_params_from_calibration_file(self, file):
        """
        Reads the saved parameters from `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`
        or uses the default values if the file doesn't exist. Adjusts the ROS parameters for the
        node with the new values.
        """

        def readFile(fname):
            with open(fname, "r") as in_file:
                try:
                    return yaml.load(in_file, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    self.logfatal("YAML syntax error. File: %s fname. Exc: %s" % (fname, exc))
                    return None

        # Check file existence
        cali_file_folder = os.path.join("/data/config/calibrations", file)
        fname = os.path.join(cali_file_folder, self.veh + ".yaml")
        # Use the default values from the config folder if a robot-specific file does not exist.
        if not os.path.isfile(fname):
            fname = os.path.join(cali_file_folder, "default.yaml")
            self.logwarn("Kinematic calibration %s not found! Using default instead." % fname)
            return readFile(fname)
        else:
            return readFile(fname)


if __name__ == "__main__":
    # Initialize the node
    dino_segmentation_node = ObjectDetectionNode(node_name='dino_segmentation_node')
    # Keep it spinning
    rospy.spin()
