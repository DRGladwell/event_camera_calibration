import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

''' clips to read '''
event_clip = cv2.VideoCapture('Events_100X_BF_10umRed_13Jan20_bead1_1_td.avi')
#event_clip = cv2.VideoCapture('calibration_clip.avi')

''' variable used to name frames '''
number_frames = 1000000

''' constant used to set how regularly a frame is captured '''
WAIT_TIME = 0.16

''' the higher the number the more frames are skipped in the event camera feed '''
frame_speed = 5

''' thread for displaying result'''
clean_thread = None

''' clean images '''
clean_frames = []

''' Generate grey image '''
# Create a blank 300x300 black image
image_grey = np.zeros((429, 544, 1), np.uint8)
# Fill image with grey
image_grey[:] = (125)

# method to read video feed and extract frames
def clean_up_clip():
    while(True):
        ''' read event_clip  '''
        ret, frame = event_clip.read()

        ''' convert clip to grayscale '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ''' Blur Image to eliminate noise '''
        blurred = cv2.GaussianBlur(gray, (5,5), 0)

        ''' Threshold positive event  '''
        thresh1 = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)[1]
        ''' Threshold negative event  '''
        blurred_inverse = cv2.bitwise_not(blurred)
        thresh2 = cv2.threshold(blurred_inverse, 165, 255, cv2.THRESH_BINARY)[1]

        ''' Create mask for events '''
        thresh = thresh1 + thresh2

        ''' be generous and add areas around event region '''
        thresh_dilate = cv2.dilate(thresh, None, iterations=5)

        ''' create mask for none event'''
        thresh_inverse = cv2.bitwise_not(thresh)

        ''' Generate clean images of event and none event '''
        output_event = cv2.bitwise_and(gray, gray, mask=thresh)
        output_none_event = cv2.bitwise_and(image_grey, image_grey, mask=thresh_inverse)

        ''' Generate image without any noise '''
        final_ouptut = output_event + output_none_event

        global clean_frames
        clean_frames.append(final_ouptut)


''' launches clean_up_clip '''
def launch_clean_thread(clean_thread):
    # if no running display_thread object exists, create one.
    if clean_thread is None:
        executorTeam = ThreadPoolExecutor()
        # "display_thread" is a future object. It has a couple of methods used to control the thread.
        clean_thread = executorTeam.submit(clean_up_clip)
        print("Function show_frames is running at time: " + str(int(time.time())) + " seconds.")
        # very important to stop the code from blocking in this method (blocking = wait for display_thread to complete)
        executorTeam.shutdown(wait=False)
        #thread2.add_done_callback(merge_cv_thread)

        return clean_thread


# method to save a frame
def save_frame(frame, number_frames):
    cv2.imwrite(("event_camera_frames/" + str(number_frames) + '_frame.jpg'), frame)
    number_frames += 1
    return number_frames


if __name__ == '__main__':
    # clean most of the noise from the event camera feed. Return cleaned clip in an array
    launch_clean_thread(clean_thread)
    frame_position = 0
    while(True):
        # timer start to regulate speed of code
        start = time.perf_counter()

        if len(clean_frames) > 0:
            # save the frame to file
            number_frames = save_frame(clean_frames[frame_position], number_frames)
            # show the displayFrame and check if a key is pressed
            cv2.imshow("Clean frame display", clean_frames[frame_position])
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, stop the loop
            if key == ord("q") or key == ord("Q"):
                break

        while (time.perf_counter() - start < WAIT_TIME):
            time.sleep(0.01)

        frame_position += frame_speed


