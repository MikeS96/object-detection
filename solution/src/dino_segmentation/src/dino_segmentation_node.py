#!/usr/bin/env python3

import yaml
import os

import numpy as np
import imgviz
import rospy
import rospkg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart
import cv2
from dino_segmentation.model import Wrapper
from cv_bridge import CvBridge
from integration import get_steer_matrix_left_lane_markings, get_steer_matrix_right_lane_markings, detect_lane_markings, \
    rescale
from integration import vanilla_servoing_mask, obstables_servoing_mask


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

        # The following are used for scaling
        self.steer_max = -1
        self.VLS_ACTION = None
        self.VLS_STOPPED = True

        # Get the steering gain (omega_max) from the calibration file
        # It defines the maximum omega used to scale normalized steering command
        kinematics_calib = self.read_params_from_calibration_file('kinematics')
        self.omega_max = kinematics_calib.get('omega_max', 6.0)

        # CMD publisher
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic,
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        # Episode start subscriber
        episode_start_topic = f"/{self.veh}/episode_start"
        rospy.Subscriber(episode_start_topic,
                         EpisodeStart,
                         self.cb_episode_start,
                         queue_size=1)

        # Image segmentation publisher
        self.pub_detections_image = rospy.Publisher(
            "~dino_segmentation_img", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Left mask publisher
        self.pub_left_mask = rospy.Publisher(
            "~lt_mask_seg", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Right mask publisher
        self.pub_right_mask = rospy.Publisher(
            "~rt_mask_seg", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Image subscriber
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
        self.v_0 = rospy.get_param('~speed', 0.2)
        self.veh = rospy.get_namespace().strip("/")
        aido_eval = rospy.get_param("~AIDO_eval", False)
        self.log(f"AIDO EVAL VAR: {aido_eval}")
        self.log("Starting model loading!")
        self._debug = rospy.get_param("~debug", False)
        self._avoid = rospy.get_param("~avoid", False)
        self._model_name = rospy.get_param("~model", '1_mlp_frozen_42')
        self.model_wrapper = Wrapper(self._model_name)
        self.log("Finished model loading!")
        self.first_image_received = False
        self.initialized = True
        self.log("Initialized!")

    def cb_episode_start(self, msg: EpisodeStart):
        loaded = yaml.load(msg.other_payload_yaml, Loader=yaml.FullLoader)
        if "calibration_value" in loaded:
            if self.AIDO_eval:
                self.steer_max = loaded["calibration_value"]
                # release robot
                self.VLS_ACTION = "go"
                self.VLS_STOPPED = False
                # NOTE: this is needed to trigger the agent and get another image back
                self.publish_command([0, 0])
            else:
                self.loginfo("Given calibration ignored as the test is running locally.")
        else:
            self.logwarn("No calibration value received. If you are running this on a real robot "
                         "or on local simulation you can ignore this message.")

    def image_cb(self, image_msg):
        if not self.initialized:
            self.publish_command([0, 0])
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

        img = cv2.resize(image, (480, 480))
        pred_mask, class_names = self.model_wrapper.predict(image)

        # Create binary mask out of segmentation mask
        if self._avoid:
            # self.steer_max = 3000  # Overwrite this one (Simulation)
            self.steer_max = 300
            weighted_mask = obstables_servoing_mask(pred_mask, self.class2int)
        else:
            # self.steer_max = 3000  # Overwrite this one (Simulation)
            self.steer_max = 300  # Overwrite this one (real)
            weighted_mask = vanilla_servoing_mask(pred_mask, self.class2int)

        # Retrieve masks
        left_mask, right_mask = detect_lane_markings(weighted_mask, pred_mask, self.class2int, self._avoid)

        # CMD commands computation
        # Load masks
        shape = weighted_mask.shape[0:2]
        steer_matrix_left_lm = get_steer_matrix_left_lane_markings(shape)
        steer_matrix_right_lm = get_steer_matrix_right_lane_markings(shape)

        steer = float(np.sum(left_mask * steer_matrix_left_lm)) + \
                float(np.sum(right_mask * steer_matrix_right_lm))

        # now rescale from 0 to 1
        steer_scaled = np.sign(steer) * rescale(min(np.abs(steer), self.steer_max), 0, self.steer_max)

        # Hand stop
        hands_patch = (pred_mask == self.class2int['hand']).sum()

        v = 0 if hands_patch > 10000 else self.v_0

        u = [v, steer_scaled * self.omega_max]
        self.publish_command(u)

        # self.logging to screen for debugging purposes
        self.log("    VISUAL SERVOING    ")
        self.log(f"Steering: (Unnormalized) : {int(steer)} / {int(self.steer_max)},"
                 f"  Steering (Normalized) : {np.round(steer_scaled, 1)}")
        self.log(f"Command v : {np.round(u[0], 2)},  omega : {np.round(u[1], 2)}")

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

            lt_mask_viz = self.bridge.cv2_to_imgmsg(lt_mask_viz, encoding="mono8")
            rt_mask_viz = self.bridge.cv2_to_imgmsg(rt_mask_viz, encoding="mono8")
            # Publish!
            self.pub_left_mask.publish(lt_mask_viz)
            self.pub_right_mask.publish(rt_mask_viz)

    def publish_command(self, u):
        """Publishes a car command message.
        Args:
            u (:obj:`tuple(double, double)`): tuple containing [v, w] for the control action.
        """

        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()

        car_control_msg.v = u[0]  # v
        car_control_msg.omega = u[1]  # omega

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
