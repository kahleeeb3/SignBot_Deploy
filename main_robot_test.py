# from env_1 import *
import cv2
from modules.libraries import *
from modules.pose_hand_landmark_code.MediapipeLandmarks import HandDetectionModel, PoseDetectionModel
from modules.ImageProcessing import *
from modules.ModelPrediction import *
# from modules.CosMosEncoding import *
import time


# if __name__ == '__main__':
#     processor = ImageProcessing()
#     model_pred = ModelPrediction(model_path=os.path.join(".", "pretrained_ckpts", 
#                                                          "ResCNNMAE_air_FT_mask_ratio_0",
#                                                          "fold_2", 
#                                                          "best_epoch.ckpt"))
#     cosmos_model = CosmosModelPrediction(model_path= os.path.join(".",
#                                                                   "pretrained_ckpts",
#                                                                   "COSMOS_trained_6_class",
#                                                                   "best_epoch.ckpt"))
#     cosmos_embedding = CosmosEmbedding()
#     frame_count = 0
#     raw_video = []
#     landmark_video = []
#     position = (50, 50)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     color = (0, 0, 255)  # Green color in BGR
#     thickness = 2
#     line_type = cv2.LINE_AA
#     # Open the default camera
#     cap = cv2.VideoCapture(1)
#     frame_number = 0
#     flag_video_saved = False
#     if not cap.isOpened():
#         print('Error in opening Camera for real-feed')
#         exit()
#     data_acquitision_flag=True
#     gesture_counter = 0
    
#     while True:
#         if data_acquitision_flag:
#             print('Be ready for the gesture number: ', gesture_counter+1)
#             print('Data acqusition will be started in 3 seconds')
#             time.sleep(3)
#             start_time, end_time = 0, 0
#             data_acquitision_flag=False
#         success, frame = cap.read()
#         #  Break the loop if the video is finished
#         if success:
#             height, width, channel = frame.shape
            
#             frame = cv2.resize(frame, (640, 480))
#             cv2.imshow('image', frame)
            
#             processor.get_annotated_image(frame)
#             if frame_count == 0:
#                 if processor.annotated_image.shape[0] !=0:
#                     text_frame = np.ones((20,60,3))
#                     cv2.putText(text_frame, 'Video Recording Just Started', position, font, font_scale, color, thickness, line_type)
#                     # Display the resulting frame
#                     cv2.imshow('Text', text_frame)
#                     cv2.imshow('landamrk', processor.annotated_image)
#                     cv2.waitKey(1)
#                     raw_video.append(cv2.resize(frame, (224, 224)).astype(np.uint8))
#                     landmark_video.append(cv2.resize(processor.annotated_image, (224, 224)).astype(np.uint8))
#                     frame_count += 1
#                     start_time = time.time() 
#                 else:
#                     # Add text to the frame
#                     text_frame = np.ones((20,60,3))
#                     cv2.putText(text_frame, 'No Data Acquisition', position, font, font_scale, color, thickness, line_type)
#                     # Display the resulting frame
#                     cv2.imshow('Text', text_frame)
#                     cv2.waitKey(1)
#                     continue
#             elif frame_count>0:
#                 if processor.annotated_image.shape[0] !=0:
#                     # Add text to the frame
#                     text_frame = np.ones((20,60,3))
#                     cv2.putText(text_frame, 'Video Recorging Going on', position, font, font_scale, color, thickness, line_type)
#                     # Display the resulting frame
#                     cv2.imshow('Text', text_frame)
#                     cv2.waitKey(1)

#                     cv2.imshow('landamrk', processor.annotated_image)
#                     cv2.waitKey(1)
#                     raw_video.append(cv2.resize(frame, (224, 224)).astype(np.uint8))
#                     landmark_video.append(cv2.resize(processor.annotated_image, (224, 224)).astype(np.uint8))
#                     frame_count += 1
#                     end_time = time.time()
#                 else:
#                     elapsed_time = end_time - start_time

#                     if elapsed_time < 3:
#                         continue
#                     else:
#                         # Add text to the frame
#                         text_frame = np.ones((20,60,3))
#                         cv2.putText(text_frame, 'Video Recording Just Stopped', position, font, font_scale, color, thickness, line_type)
#                         # Display the resulting frame
#                         cv2.imshow('Text', text_frame)
#                         cv2.waitKey(1)
                        
#                         raw_video = np.array(raw_video)
#                         landmark_video = np.array(landmark_video)
#                         # print('Raw Video shape: ', raw_video.shape)
#                         # print('Landmark Video Shape: ', landmark_video.shape)
#                         raw_latent = cosmos_embedding.generate_embedding(input_video=raw_video)
#                         landmark_latent = cosmos_embedding.generate_embedding(input_video=landmark_video)
#                         # print('Raw latent shape: ', raw_latent.shape)
#                         # print('Landmark latent Shape: ', landmark_latent.shape)
#                         cv2.destroyWindow('landamrk')

#                         # Model 1 Prediction
#                         direction = model_pred.prediction(raw_video=raw_video,
#                                                         landmark_video=landmark_video)
#                         print(f"#########  {direction}   ############")

#                         # Model 2 (Cosmos Prediction)
#                         direction_2 = cosmos_model.prediction(raw_latent, landmark_latent)
#                         print(f"#########  {direction_2}   ############")
#                         # Add text to the frame
#                         text_frame = np.ones((200,200,3))
#                         cv2.putText(text_frame, direction, position, font, font_scale, color, thickness, line_type)
#                         # Display the resulting frame
#                         cv2.imshow('Text', text_frame)
#                         cv2.waitKey(5)
#                         # env.move_robot(direction)
#                         frame_count = 0
#                         raw_video = []
#                         landmark_video = []

#             if cv2.waitKey(1) == ord('q'):
#                 cv2.destroyAllWindows()
#                 break


if __name__ == '__main__':
    # Initialize components
    processor = ImageProcessing()
    model_pred = ModelPrediction(model_path=os.path.join(
        ".", "modules", "pretrained_ckpts", "ResCNNMAE_air_FT_mask_ratio_0", "fold_2", "best_epoch.ckpt"))
    # cosmos_model = CosmosModelPrediction(model_path=os.path.join(
    #     ".", "pretrained_ckpts", "Cosmos_V2_own_6class_natops", "fold_1",  "best_epoch.ckpt"))
    # cosmos_embedding = CosmosEmbedding()

    # Setup parameters
    frame_count = 0
    raw_video, landmark_video = [], []
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # Red in BGR
    thickness = 2
    line_type = cv2.LINE_AA

    # Camera init
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error in opening Camera for real-feed')
        exit()

    gesture_counter = 0
    data_acquisition_flag = True

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

        # Resize and show the frame
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('image', frame)

        processor.get_annotated_image(frame)

        if frame_count == 0:
            if processor.annotated_image.shape[0] != 0:
                # Show "Recording Started" message
                text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                cv2.putText(text_frame, 'Video Recording Just Started', position, font, font_scale, color, thickness, line_type)
                cv2.imshow('Text', text_frame)
                cv2.imshow('landmark', processor.annotated_image)
                cv2.waitKey(1)

                # Save first frame
                raw_video.append(cv2.resize(frame, (224, 224)).astype(np.uint8))
                landmark_video.append(cv2.resize(processor.annotated_image, (224, 224)).astype(np.uint8))
                frame_count += 1
                start_time = time.time()
            else:
                # No data available yet
                text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                cv2.putText(text_frame, 'No Data Acquisition', position, font, font_scale, color, thickness, line_type)
                cv2.imshow('Text', text_frame)
                cv2.waitKey(1)
                continue

        else:
            if processor.annotated_image.shape[0] != 0:
                # Continue recording
                text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                cv2.putText(text_frame, 'Video Recording Going On', position, font, font_scale, color, thickness, line_type)
                cv2.imshow('Text', text_frame)
                cv2.imshow('landmark', processor.annotated_image)
                cv2.waitKey(1)

                raw_video.append(cv2.resize(frame, (224, 224)).astype(np.uint8))
                landmark_video.append(cv2.resize(processor.annotated_image, (224, 224)).astype(np.uint8))
                frame_count += 1
                end_time = time.time()
            else:
                elapsed_time = end_time - start_time
                if elapsed_time < 3:
                    continue

                # End of recording
                text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                cv2.putText(text_frame, 'Video Recording Just Stopped', position, font, font_scale, color, thickness, line_type)
                cv2.imshow('Text', text_frame)
                cv2.waitKey(1)

                # Convert to arrays
                raw_video_np = np.array(raw_video)
                landmark_video_np = np.array(landmark_video)
                print(raw_video_np.shape)

                # # Generate embeddings
                # raw_latent = cosmos_embedding.generate_embedding(raw_video_np[..., ::-1])
                # landmark_latent = cosmos_embedding.generate_embedding(landmark_video_np[..., ::-1])

                cv2.destroyWindow('landmark')

                # Model 1 Prediction
                direction = model_pred.prediction(raw_video_np, landmark_video_np)
                print(f"######### Model 1 Prediction: {direction} #########")

                # # Model 2 Prediction
                # direction_2 = cosmos_model.prediction(raw_latent, landmark_latent)
                # print(f"######### Model 2 Prediction: {direction_2} #########")

                # Display result
                text_frame = np.ones((200, 600, 3), dtype=np.uint8) * 255
                cv2.putText(text_frame, direction, position, font, font_scale, color, thickness, line_type)
                cv2.imshow('Text', text_frame)
                cv2.waitKey(5)

                # Reset state
                frame_count = 0
                raw_video.clear()
                landmark_video.clear()
                data_acquisition_flag = True
                gesture_counter += 1
                # del raw_latent, landmark_latent,
                del raw_video_np, landmark_video_np

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()