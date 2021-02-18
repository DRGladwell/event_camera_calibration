import imutils
import numpy as np
import cv2
import time

#cap = cv2.VideoCapture('Events_100X_BF_10umRed_13Jan20_bead1_1_td.avi')
#cap = cv2.VideoCapture('Events_100X_BF_10umRed_13Jan20_bead1_2_td.avi')
cap = cv2.VideoCapture('Events_100X_BF_10umRed_13Jan20_bead1_3_td.avi')

#cap = cv2.VideoCapture('Events_100X_BF_Leish_MG_13Jan20_td.avi')
#cap = cv2.VideoCapture('Events_100X_BF_Leish_MG_13Jan20_3_td.avi')
#cap = cv2.VideoCapture('Events_100X_BF_Leish_MG_13Jan20_2_td.avi')

global number_pictures
number_pictures = 1000283  #Used a high number to avoid ordering issues

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




def extract_roi(box, gray):
    (x, y, h, w) = box
    # make the side of square roi equal to largest length of bounding box
    size_of_square_ROI = h if h > w else w
    # find the center of bounding box
    center_y = y + h / 2
    center_x = x + w / 2
    # find the top left corner
    top_x = int(round(center_x - size_of_square_ROI / 2))
    top_y = int(round(center_y - size_of_square_ROI / 2))

    # find right and bottom most position
    right_most = top_x + size_of_square_ROI
    bottom_most = top_y + size_of_square_ROI

    if right_most > gray.shape[1] or bottom_most > gray.shape[0] or top_x < 0 or top_y < 0:
        print("[INFO] ROI out of bounds")
    else:
        extracted_roi = gray[top_y:bottom_most, top_x:right_most]
        cv2.imwrite(("bead_samples/" + str(number_pictures) + '_bead1.jpg'), extracted_roi)
        number_pictures += 1
        print(number_pictures)


while(cap.isOpened()):
    #time.sleep(0.032)
    start_time = time.time()
    ret, frame = cap.read()
    # frm = np.asarray(frame)
    # print(np.shape(frm))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #sharpened = unsharp_mask(gray)

    ''' Blur Image to eliminate noise '''
    blurred = cv2.GaussianBlur(gray, (9,9), 0)

    ''' Threshold positive event  '''
    thresh1 = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)[1]
    ''' Threshold negative event  '''
    blurred_inverse = cv2.bitwise_not(blurred)
    thresh2 = cv2.threshold(blurred_inverse, 170, 255, cv2.THRESH_BINARY)[1]

    ''' Create mask for events '''
    thresh = thresh1 + thresh2
    # perform a blur filter to connect blobs of events that are close to each other
    kernel = np.ones((11, 11), np.float32) / 25
    filtered = cv2.filter2D(thresh, -1, kernel)
    thresh_region = cv2.threshold(filtered, 1, 255, cv2.THRESH_BINARY)[1]

    ''' create mask for none event'''
    thresh_inverse = cv2.bitwise_not(thresh_region)

    ''' Generate clean images of event and none event '''
    output_event = cv2.bitwise_and(gray, gray, mask=thresh)
    output_none_event = cv2.bitwise_and(image_grey, image_grey, mask=thresh_inverse)

    ''' Generate image without any noise '''
    final_ouptut = output_event + output_none_event

############
    #blurred = cv2.GaussianBlur(thresh_region, (11, 11), 0)
    #thresh_erode = cv2.erode(thresh_region, None, iterations=5)
    thresh_dilate = cv2.dilate(thresh_region, None, iterations=5)
    thresh_erode = cv2.erode(thresh_dilate, None, iterations=7)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh_erode.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        #cv2.drawContours(final_ouptut, [c], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(c)



        if (h > 429/10 and w > 544/10):
            ''' uncomment to extract region of interest, change method
                definition to extract to new file.       
            '''
            #extract_roi(cv2.boundingRect(c), gray)

            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(gray, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(gray, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #print("--- %s seconds ---" % (time.time() - start_time))
cap.release()
cv2.destroyAllWindows()