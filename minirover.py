import numpy as np
import cv2
import time
import matplotlib.pyplot as plt 

def region_of_interest(image):
    squares = np.array([
        [(0,200),(0,300),(300,200)],
        [(0,300),(300,200),(300,300)],
        [(500,200),(500,300),(900,200)],
        [(500,300),(900,300),(900,200)],
        ])

    mask1 = np.zeros_like(image)
    cv2.fillPoly(mask1,squares,255)
    masked_image = cv2.bitwise_and(image,mask1)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

def main():
    video_capture = cv2.VideoCapture('/home/shervin/Downloads/Minirover.mp4')
    while(True):
        ret, frame = video_capture.read()
        cv2.imshow("Frame",frame)
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blueLower =(110, 60, 1)
        blueUpper = (179, 255, 100)
        mask = cv2.inRange(frame1, blueLower, blueUpper)
        cv2.imshow("mask",mask)

        new_image = np.copy(frame)
        gray_image = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)

        canny = cv2.Canny(mask,50,250)
        cv2.imshow("canny",canny)
        cropped_image = region_of_interest(canny)
        cv2.imshow("cropped image",cropped_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=40)

        line_image = display_lines(new_image, lines)
        cv2.imshow("line image",line_image)

        combo_image = cv2.addWeighted(new_image, 0.8, line_image, 1, 1)
        cv2.imshow("combo image",combo_image)


        
        time.sleep(0.02)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

