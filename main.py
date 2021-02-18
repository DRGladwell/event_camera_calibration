'''
Code to simultaneously flash a grid pattern and capture event camera images.
The reason for integrating these two tasks together is to make sure the camera
correctly captures the events generated. The code will make an attempt to clean
up the event image without mutating the pixels generated.
'''

import cv2
import numpy as np
import time


# method to display the image of a grid on the screen
def display_calibration_pattern():
    calib_grid_img = cv2.imread("calibration_grid.png")
    cv2.imshow("calib_grid_img", calib_grid_img)
    cv2.waitKey(0)
    cv2.destroyWindow("calib_grid_img")

# method to flash the image of a grid on the screen
def flash_calibration_pattern():
    calib_grid_img = cv2.imread("calibration_grid.png")
    calib_grid_img = cv2.cvtColor(calib_grid_img, cv2.COLOR_BGR2GRAY)
    (height,width) = calib_grid_img.shape
    print(height)
    print(width)

    # flash white
    img = np.ones([height, width], dtype=np.uint8) * 255
    # flash black
    img = np.zeros([height, width], dtype=np.uint8)

    control_int = 0
    while (True):
        start = time.perf_counter()
        if (control_int == 0):
            cv2.imshow("img", img)
            control_int = 1
            while (time.perf_counter() - start < 0.016):
                time.sleep(0.00001)
            print(time.perf_counter() - start)
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q") or key == ord("Q"):
                break

        else:
            cv2.imshow("img", calib_grid_img)
            control_int = 0
            while (time.perf_counter() - start < 0.016):
                time.sleep(0.00001)
            print(time.perf_counter() - start)
            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q") or key == ord("Q"):
                break


if __name__ == '__main__':
    display_calibration_pattern()
    flash_calibration_pattern()



