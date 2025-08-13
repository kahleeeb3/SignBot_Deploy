

#To run, user will need to install Boston Dynamics SDK to SignBot_Deploy environment:

# pip install bosdyn-client==4.1.1 bosdyn-mission==4.1.1 bosdyn-choreography-client==4.1.1 bosdyn-orbit==4.1.1


import cv2
import numpy as np
import time
import sys
import os

from libraries import *
from pose_hand_landmark_code.MediapipeLandmarks import HandDetectionModel, PoseDetectionModel
from ImageProcessing import *
from ModelPrediction import *

from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    blocking_stand,
)

if __name__ == '__main__':
    # Initialize gesture processing
    processor = ImageProcessing()
    model_pred = ModelPrediction(model_path=os.path.join(
        ".", "pretrained_ckpts", "RESCNNMAE_air_FT_mask_ratio_0", "fold_2", "best_epoch.ckpt"))

    frame_count = 0
    raw_video, landmark_video = [], []
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2
    line_type = cv2.LINE_AA

    # Camera init
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('Error in opening Camera for real-feed')
        exit()

    gesture_counter = 0
    data_acquisition_flag = True

    # Mac initialization
    ROBOT_IP = "192.168.80.3"
    ROBOT_USERNAME = "admin"
    ROBOT_PASSWORD = "ccr2xy2brw6n"

    sdk = create_standard_sdk("SpotController")
    robot = sdk.create_robot(ROBOT_IP)
    robot.authenticate(ROBOT_USERNAME, ROBOT_PASSWORD)
    robot.time_sync.wait_for_sync()

    if robot.is_estopped():
        print("Robot is estopped. Clear E-Stop before running.")
        sys.exit(1)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease = lease_client.acquire()   

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot.power_on(timeout_sec=20)
        blocking_stand(command_client, timeout_sec=10)

        def move_spot(direction, command_client):
            MOVE_DURATION = 2.0 #This is in seconds
            end_time_secs = time.time() + MOVE_DURATION

            if direction == "left":
                print("Spot sidestepping left.")
                traj = RobotCommandBuilder.synchro_velocity_command(
                    v_x=0.0, v_y=0.2, v_rot=0.0
                )
            elif direction == "right":
                print("Spot sidestepping right.")
                traj = RobotCommandBuilder.synchro_velocity_command(
                    v_x=0.0, v_y=-0.2, v_rot=0.0
                )
            elif direction == "forward":
                print("Spot walking forward.")
                traj = RobotCommandBuilder.synchro_velocity_command(
                    v_x=0.2, v_y=0.0, v_rot=0.0
                )
            else:
                print("Unknown direction:", direction)
                return

            # Send the velocity command with an expiration time
            command_client.robot_command(command=traj,
                                        end_time_secs=end_time_secs)
            time.sleep(MOVE_DURATION)

            # Then immediately send a stop
            stop = RobotCommandBuilder.stop_command()
            command_client.robot_command(stop)

        # While loop for gesture capture
        while True:
            if data_acquisition_flag:
                print(f'Be ready for the gesture number: {gesture_counter + 1}')
                print('Data acquisition will be started in 3 seconds')
                time.sleep(3)
                start_time, end_time = 0, 0
                data_acquisition_flag = False

            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 480))
            cv2.imshow('image', frame)

            processor.get_annotated_image(frame)

            if frame_count == 0:
                if processor.annotated_image.shape[0] != 0:
                    text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                    cv2.putText(text_frame, 'Video Recording Just Started', position,
                                font, font_scale, color, thickness, line_type)
                    cv2.imshow('Text', text_frame)
                    cv2.imshow('landmark', processor.annotated_image)
                    cv2.waitKey(1)

                    raw_video.append(cv2.resize(frame, (224, 224)).astype(np.uint8))
                    landmark_video.append(cv2.resize(processor.annotated_image,
                                                    (224, 224)).astype(np.uint8))
                    frame_count += 1
                    start_time = time.time()
                else:
                    text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                    cv2.putText(text_frame, 'No Data Acquisition', position,
                                font, font_scale, color, thickness, line_type)
                    cv2.imshow('Text', text_frame)
                    cv2.waitKey(1)
                    continue
            else:
                if processor.annotated_image.shape[0] != 0:
                    text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                    cv2.putText(text_frame, 'Video Recording Going On', position,
                                font, font_scale, color, thickness, line_type)
                    cv2.imshow('Text', text_frame)
                    cv2.imshow('landmark', processor.annotated_image)
                    cv2.waitKey(1)

                    raw_video.append(cv2.resize(frame, (224, 224)).astype(np.uint8))
                    landmark_video.append(cv2.resize(processor.annotated_image,
                                                    (224, 224)).astype(np.uint8))
                    frame_count += 1
                    end_time = time.time()
                else:
                    elapsed_time = end_time - start_time
                    if elapsed_time < 3:
                        continue

                    text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                    cv2.putText(text_frame, 'Video Recording Just Stopped', position,
                                font, font_scale, color, thickness, line_type)
                    cv2.imshow('Text', text_frame)
                    cv2.waitKey(1)

                    raw_video_np = np.array(raw_video)
                    landmark_video_np = np.array(landmark_video)
                    print(raw_video_np.shape)

                    cv2.destroyWindow('landmark')

                    # Model 1 Prediction
                    direction = model_pred.prediction(raw_video_np, landmark_video_np)
                    print(f"######### Model 1 Prediction: {direction} #########")
                    sys.stdout.flush()

                    # Move Mac
                    if direction in ["left", "right", "forward"]:
                        move_spot(direction, command_client)

                    text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                    cv2.putText(text_frame, direction, position,
                                font, font_scale, color, thickness, line_type)
                    cv2.imshow('Text', text_frame)
                    cv2.waitKey(5)

                    frame_count = 0
                    raw_video.clear()
                    landmark_video.clear()
                    data_acquisition_flag = True
                    gesture_counter += 1
                    del raw_video_np, landmark_video_np

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        robot.power_off(cut_immediately=False)
        print("Mac powered off.")



