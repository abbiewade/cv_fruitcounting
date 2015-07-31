import numpy as np
import sys
import cv2

class Fruit_detector():
    def __init__(self, file):
        #Stores the video file
        self.video = cv2.VideoCapture(file)
        #Stores the first frame of the video
        self.staticFrame = self.video.read(0)[1]
        #Stores the current frame of the video
        self.frame = None
        #Stores whether to draw an outline
        self.outline = False
        #Stores whether to draw the labels
        self.realtime_label = True
        #Stores whether to draw the count
        self.display_count = True
        #Stores the count
        self.count = np.zeros((5,1))
        #The list of objects which have recently been counted
        self.object_locations = []
        #The counting lines y centre
        self.line_loc = 500
        #The line size
        self.line_size = 15
        #The time (In frames) until object locations are removed from self.object_locations
        self.dequeue_timer = 10
        #The range of xvalues to count as the same for the object locations
        self.xbounds = 15

    def detect(self):
        speed = 15
        while True:
            ret, self.frame = self.video.read(0)
            if ret:
                ###
                #Perform preprocessing and then classification
                ###
                transformed_frame = self.preprocess()
                self.recognise(transformed_frame)

                ###
                #If currently drawing a bounding box then draw or if drawing count
                ###

                if self.outline:
                    self.draw_bounding()
                if self.display_count:
                    self.draw_count()

                cv2.imshow("Results", self.frame)

                ###
                #Decrement the clip timer on the approximate count
                ###

                for i in self.object_locations:
                    i[1] -= 1
                    if i[1] <= 0:
                        self.object_locations.remove(i) 
                
                ###
                #Check if a key has been pressed and add a delay to the clip
                ###

                key = cv2.waitKey(speed)
                if key == ord('l'):
                    self.realtime_label = not self.realtime_label
                elif key == ord('c'):
                    self.display_count = not self.display_count
                elif key == ord('b'):
                    self.outline = not self.outline
                elif key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
                elif key == ord('s'):
                    speed += 100
                elif key == ord('f'):
                    speed -= 100
                    if speed <= 5:
                        speed = 5
            else:
                #If the clip is done or failed to load break 
                break
        #Print the final fruit count
        print(self.count)

    def preprocess(self):
        ##
        # REMOVE THE STATICFRAMES ELEMENTS FROM THE CURRENT FRAME AND CONVERT TO HSV
        ##

        difference_frame = cv2.absdiff(self.staticFrame, self.frame)
        hsv = cv2.cvtColor(difference_frame, cv2.COLOR_BGR2HSV)
        frame = hsv[:,:,2]

        ##
        # REMOVING NOISE FROM IMAGE
        ##
        noise_ind = frame < 80
        frame[noise_ind] = 0
        frame = cv2.medianBlur(frame, 5)

        ##
        # MAKING IMAGE BINARY
        ##
        noise_ind = frame > 0
        frame[noise_ind] = 255

        ##
        # WATERSHED FILTERING (Remove bridges)
        ##

        dist_transform = cv2.distanceTransform(frame, cv2.cv.CV_DIST_L2,5) 
        mean_bridge = np.median(dist_transform[np.nonzero(dist_transform)])
        _, noiseless_frame = cv2.threshold(dist_transform, 1.50*mean_bridge, 255,0)


        ##
        # DILATE TO REMOVE SMALL GAPS IN OBJECTS
        ##

        noiseless_frame = np.uint8(noiseless_frame)
        noiseless_frame = cv2.dilate(noiseless_frame, (3,3), iterations = 14)

        return noiseless_frame

    def recognise(self, transformed_frame):
        ##
        # COMPUTE THE CONTOURS FOR THE IMAGE
        ##

        contours, hierarchy = cv2.findContours(transformed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #Sanity check 
        if contours is not None:
            for cnt in contours:
                #Remove contours which are too large or too small to be fruit
                if (cnt.size > 20) and (cnt.size < 800): 
                    ###
                    #Calculate useful values
                    ###

                    #Find the centre of the contour
                    m  = cv2.moments(cnt)
                    x_c = int(round(m['m10']/m['m00']))
                    y_c = int(round(m['m01']/m['m00']))

                    #The bounds of our counting rectangle
                    max_b = self.line_loc + self.line_size
                    min_b = self.line_loc - self.line_size

                    #Get the area and extent of the contour
                    area = cv2.contourArea(cnt)
                    x,y,w,h = cv2.boundingRect(cnt)
                    rect_area = w*h
                    extent = float(area)/rect_area

                    #Get the mean HSV and BGR values
                    mask = np.zeros(self.frame[:,:,0].shape, np.uint8)
                    cv2.drawContours(mask,[cnt],0,255,-1)
                    mean_colour = cv2.mean(self.frame, mask = mask)
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    mean_hue = cv2.mean(hsv, mask = mask)

                    ###
                    #Check if the contours centre is in our counting rectangle and if we have seen it before
                    ###

                    flag = False
                    #Check if over in rectangle
                    if  (y_c >= min_b) and (y_c <= max_b):
                        flag = True
                        #Check if seen before
                        for (a,_) in self.object_locations:
                            if a > x_c - self.xbounds and a < x_c + self.xbounds:
                                flag = False
                        if flag:
                            #Change color of circle drawn on contour centre
                            self.object_locations.append([x_c, self.dequeue_timer])
                            cv2.circle(self.frame,(x_c,y_c),5,(255,0,0), -1)
                    else:
                        cv2.circle(self.frame,(x_c,y_c),5,(0,255,0), -1)

                    ###
                    #Labeling
                    ###
                    #Order: Banana, Tomato, Pear, Orange, Else
                    #Note: Count is stored in self.count[]

                    if extent < 0.5:
                        cv2.putText(self.frame,"banana", (x_c-10,y_c+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
                        if flag:
                            self.count[0] += 1
                    
                    elif mean_colour[0] < 100 and mean_colour[2] > 230 and mean_hue[0] < 30 and mean_colour[1]  < 220 and mean_hue[1] > 160 and cnt.size < 190:#
                        cv2.putText(self.frame,"tomato", (x_c-10,y_c+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
                        if flag:
                            self.count[2] += 1
                    
                    elif mean_hue[0] > 30 and mean_colour[1] > 220 and mean_colour[0] < 100 and mean_hue[1] < 200 and mean_hue[1] > 135 and mean_hue[2] > 208: 
                        cv2.putText(self.frame,"pear", (x_c-10,y_c+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
                        if flag:
                            self.count[3] += 1
                    
                    elif mean_colour[0] < 75 and mean_colour[2] > 210:
                        cv2.putText(self.frame,"orange", (x_c-10,y_c+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
                        if flag:
                            self.count[1] += 1
                    
                    else:
                        cv2.putText(self.frame,"ELSE", (x_c-10,y_c+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
                        if flag:
                            self.count[4] += 1
                            #Cause beep on unix based systems (Might not work on windows)
                            print("\a")

    def draw_bounding(self):
        ###
        #Draw_bounding draws a bounding ellipse around objects
        ###
        #Note: Does not use watershed, only removing noise.

        ###
        #Remove static frame and convert frame to HSV
        ###
        difference_frame = cv2.absdiff(self.staticFrame, self.frame)
        hsv = cv2.cvtColor(difference_frame, cv2.COLOR_BGR2HSV)
        frame = hsv[:,:,2]

        ###
        #Remove noise by soft-min
        ###
        noise_ind = frame < 100
        frame[noise_ind] = 0
        frame = cv2.medianBlur(frame, 5)   
        _, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

        ###
        #Find contours and draw ellipses over them
        ###
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if (cnt.size > 200) and (cnt.size < 500):
                ellipse = cv2.fitEllipse(cnt)
                im = cv2.ellipse(self.frame, ellipse,(0,255,0),2)   

    def draw_count(self):
        ###
        #Draw_count is used to draw the status of the fruit detection to the frame
        ###

        #Draw black box in the top left hand corner of the screen
        cv2.rectangle(self.frame, (0,0), (160,120), (0,0,0), -1)
        #Draw line for where fruit are counted
        cv2.line(self.frame, (0,self.line_loc), (self.frame.shape[1], self.line_loc), (0,0,255))

        #Write the current count of the fruit to the frame
        cv2.putText(self.frame,"Banana: " + str(self.count[0,0]), (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
        cv2.putText(self.frame,"Orange: " + str(self.count[1,0]), (0,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
        cv2.putText(self.frame,"Tomato: " + str(self.count[2,0]), (0,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
        cv2.putText(self.frame,"Pear: " + str(self.count[3,0]), (0,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
        cv2.putText(self.frame,"Apple: " + str(0.0), (0,95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
        cv2.putText(self.frame,"Other: " + str(self.count[4,0]), (0,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)

###
#The part of the code which runs the fruit counting
###

if __name__ == '__main__':
    #Check if the correct number of command line arguements were supplied
    if len(sys.argv) != 2:
        raise Exception("Incorrect number of arguements provided. \n\nUse form: python final.py path/to/clip")

    #Create the fruit detector object with the clip given
    fruit_det = Fruit_detector(sys.argv[1])

    #Run the detection pipeline
    fruit_det.detect()
