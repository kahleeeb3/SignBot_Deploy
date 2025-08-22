
from modules.ImageProcessing import *
from modules.ModelPrediction import *
from modules.signBotHelper import WebServer, ClipBuffer, FrameCapture
import time
    
# object defines
server = WebServer(port=8765, jpeg_quality=80)
clip = ClipBuffer()
camera = FrameCapture()
processor = ImageProcessing()
model_path = "./modules/pretrained_ckpts/ResCNNMAE_air_FT_mask_ratio_0/fold_2/best_epoch.ckpt"
model_pred = ModelPrediction(model_path)

# globals
frame_count = 0
acquiring_data = True
gesture_counter = 0

# run the main loop
camera.open()
server.start()

while True:

    # warn about data acquisition
    if acquiring_data:
        # server.showText(f'Be ready for the gesture number: {gesture_counter + 1}')
        # server.showText('Data acquisition will be started in 3 seconds')
        time.sleep(3)
        start_time, end_time = 0, 0
        acquiring_data = False

    # capture & show frame
    camera.capture()
    if camera.frame is None:
        break
    server.stream(camera.frame, "video0")

    # process frame
    processor.get_annotated_image(camera.frame) # process frame
    annotated = getattr(processor, "annotated_image", None) # get attr
    landmarks = isinstance(annotated, np.ndarray) and annotated.size > 0 # check type & size
    state = (frame_count == 0, landmarks) # state of the frame

    # decide state     
    if state == (True, True):
        server.showText("Video Recording Just Started")
        server.stream(processor.annotated_image, "video1")
        clip.add(camera.frame, processor.annotated_image)
        frame_count += 1
        start_time = time.time()
    elif state == (True, False):
        # server.showText("No Data Acquisition")
        pass
    elif state == (False, True):
        server.showText("Video Recording Going On")
        server.stream(processor.annotated_image, "video1")
        clip.add(camera.frame, processor.annotated_image)
        frame_count += 1
        end_time = time.time()
    else:
        elapsed_time = end_time - start_time
        if elapsed_time < 3:
            continue
        server.showText("Video Recording Just Stopped'")
        raw_np, land_np = clip.finalize()
        if raw_np is not None:
            direction = model_pred.prediction(raw_np, land_np)
            server.showText(f"Prediction: {direction}")
        frame_count = 0
        acquiring_data = True
        gesture_counter += 1        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()