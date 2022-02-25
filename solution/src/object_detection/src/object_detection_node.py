#!/usr/bin/env python3
import numpy as np
import imgviz
import rospy
import rospkg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart
import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge
from integration import NUMBER_FRAMES_SKIPPED, filter_by_classes, filter_by_bboxes, filter_by_scores


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
            "~object_detections_img", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
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
        image = cv2.resize(image, (480, 480))
        pred_mask, class_names = self.model_wrapper.predict(image)

        # Resize the original image and the predictions to 480 x 480
        img = cv2.resize(np.array(image), (480, 480))
        pred_mask = np.kron(pred_mask, np.ones((8, 8))).astype(int)  # Upscale the predictions back to 480x480

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

            obj_det_img = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            self.pub_detections_image.publish(obj_det_img)

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


if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name='object_detection_node')
    # Keep it spinning
    rospy.spin()
