import mediapipe as mp
from modules.pose_hand_landmark_code.MediapipeLandmarks import HandDetectionModel, PoseDetectionModel
import numpy as np
import os
class ImageProcessing:
    def __init__(self):
        """
        Initializes hand and pose models, and prepares internal flags.
        """
        self.__init__models() 
        self.hand_below_flag = False
        self.annotated_image = np.array([])

    def __init__models(self):
        """
        Loads the hand and pose detection models with preset configurations.
        """
        self.hand_model = HandDetectionModel(model_asset_path = os.path.join(".", "modules", "pose_hand_landmark_code", "hand_landmarker.task"),
                                           
                                min_hand_detection_confidence= 0.5,
                                min_hand_presence_confidence= 0.5)

        self.pose_model = PoseDetectionModel(model_asset_path = os.path.join(".", "modules", "pose_hand_landmark_code", "pose_landmarker_heavy.task"),
                                            
                                min_pose_detection_confidence=0.5,
                                min_pose_presence_confidence=0.5,
                                pose_coonection='upper_body')
        
    def get_results(self, image):
        """
        Runs hand and pose detection on the input image.

        Args:
            image (np.ndarray): Input image in [H, W, C], pixel range [0, 1].

        Returns:
            tuple: (hand landmarks, pose landmarks)
        """
        # Convert image to uint8 and format for MediaPipe
        image = (image * 255).astype(np.uint8)
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # get the detected landmarks
        landmark_result = self.hand_model.landmark_detection(image_mp)
        
        # get pose landmarks
        pose_landmarks_results = self.pose_model.landmark_detection(image_mp)
        return landmark_result, pose_landmarks_results
    
    def get_reference_y(self, pose_landmarks_results):
        """
        Computes average Y-values of shoulder and hip joints to estimate waist position.

        Args:
            pose_landmarks: Pose detection result from model.

        Returns:
            tuple: (ref_y1, ref_y2) or (None, None) if no pose detected
        """
        if not len(pose_landmarks_results.pose_landmarks)==0:
                   
            # ref_y1 = (pose_landmarks_results.pose_landmarks[0][12].y + pose_landmarks_results.pose_landmarks[0][24].y)/2    # right shoulder + right hip
            # ref_y2 = (pose_landmarks_results.pose_landmarks[0][11].y + pose_landmarks_results.pose_landmarks[0][23].y)/2    # left shoulder + left hip
            ref_y1 = 0.8*(pose_landmarks_results.pose_landmarks[0][24].y)   # right hip
            ref_y2 = 0.8*(pose_landmarks_results.pose_landmarks[0][23].y)    # left hip
            
        
        else: 
            ref_y1, ref_y2 = None, None
        return ref_y1, ref_y2
    
    def get_annotated_image(self, image):
        """
        Annotates image with hand and pose landmarks if both hands are above waist.

        Args:
            image (np.ndarray): Input image in [H, W, C], pixel range [0, 1].

        Side Effects:
            - Updates self.annotated_image with landmarks or empty array
            - Updates self.hand_below_flag indicating hand position
        """
        landmark_result, pose_landmarks_results = self.get_results(image)
        ref_y1, ref_y2 = self.get_reference_y(pose_landmarks_results)
        if ref_y1 != None:
            self.hand_below_flag = False
            if (pose_landmarks_results.pose_landmarks[0][16].y < ref_y1 or pose_landmarks_results.pose_landmarks[0][15].y < ref_y2):
                zero_image = image.copy()*0
                # Draw both hand and pose landmarks
                self.annotated_image = self.hand_model.draw_landmarks_on_image(zero_image, landmark_result)
                self.annotated_image = self.pose_model.draw_landmarks_on_image(self.annotated_image, pose_landmarks_results)
            else:
                self.hand_below_flag = True
                self.annotated_image = np.array([])
        else:
            # No pose detected; cannot determine hand position
            self.hand_below_flag = True

