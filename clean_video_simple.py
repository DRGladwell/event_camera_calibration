import imutils
import numpy as np
import cv2
import time
from imutils.video import WebcamVideoStream



def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=1.5):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


''' Genrate grey image '''
# Create a blank 300x300 black image
image_grey = np.zeros((429, 544, 1), np.uint8)
# Fill image with grey
image_grey[:] = (125)

camera = WebcamVideoStream('Events_100X_BF_10umRed_13Jan20_bead1_2_td.avi'.start())
while(camera.isOpened()):
    time.sleep(0.032)
    start_time = time.time()
    ret, frame = camera.read()


    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # #sharpened = unsharp_mask(gray)
    #
    # ''' Blur Image to eliminate noise '''
    # blurred = cv2.GaussianBlur(gray, (11,11), 0)
    #
    # ''' Threshold positive event  '''
    # thresh1 = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)[1]
    # ''' Threshold negative event  '''
    # blurred_inverse = cv2.bitwise_not(blurred)
    # thresh2 = cv2.threshold(blurred_inverse, 170, 255, cv2.THRESH_BINARY)[1]
    #
    # ''' Create mask for events '''
    # thresh = thresh1 + thresh2
    # # perform a blur filter to connect blobs of events that are close to each other
    # # kernel = np.ones((11, 11), np.float32) / 25
    # # filtered = cv2.filter2D(thresh, -1, kernel)
    # # thresh_region = cv2.threshold(filtered, 1, 255, cv2.THRESH_BINARY)[1]
    #
    # ''' create mask for none event'''
    # thresh_inverse = cv2.bitwise_not(thresh)
    #
    # ''' Generate clean images of event and none event '''
    # output_event = cv2.bitwise_and(gray, gray, mask=thresh)
    # output_none_event = cv2.bitwise_and(image_grey, image_grey, mask=thresh_inverse)
    #
    # ''' Generate image without any noise '''
    # final_ouptut = output_event + output_none_event

    # # find contours in the thresholded image
    # cnts = cv2.findContours(final_ouptut.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # # loop over the contours
    # for c in cnts:
    #     # compute the center of the contour
    #     M = cv2.moments(c)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #
    #     # draw the contour and center of the shape on the image
    #     #cv2.drawContours(final_ouptut, [c], -1, (0, 255, 0), 2)
    #     x, y, w, h = cv2.boundingRect(c)
    #     if (h > 429/20 and w > 544/20):
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
    #         cv2.putText(frame, "center", (cX - 20, cY - 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("--- %s seconds ---" % (time.time() - start_time))
cap.release()
cv2.destroyAllWindows()