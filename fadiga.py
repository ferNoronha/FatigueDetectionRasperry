# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

#reference: https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/

def euclidean_dist(ptA, ptB):
	# compute and return the euclidean distance between the two
	# points
	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = euclidean_dist(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
    A = euclidean_dist(mouth[2],mouth[10])
    B = euclidean_dist(mouth[4],mouth[8])
    C = euclidean_dist(mouth[0],mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def distancia_nariz_queixo(nose,jaw):
    x,y = nose[3]
    xx,yy = nose[6]
    ponto = (xx,y)
    
    C = euclidean_dist(jaw[1],jaw[15])
    A = euclidean_dist(nose[6],ponto)
    #print(nose[6])
    #print(ponto)
    
    return A

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,	help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0,	help="boolean used to indicate if TrafficHat should be used")
args = vars(ap.parse_args())
# check to see if we are using GPIO/TrafficHat as an alarm
if args["alarm"] > 0:
	from gpiozero import TrafficHat
	th = TrafficHat()
	print("[INFO] Usando traffichat...")

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 8

MOUTH_AR_THRESH = 0.890
COUNTER_MOUTH_OPEN = 0
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

COUNTER_EYE_CLOSE = 0
flag_mouth = False
flag_eye = False

FACE_AR_THRESH = 11.0

# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
print("[INFO] carregando facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(bStart, bEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(qStart, qEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

print("[INFO] iniciando video")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

array = []
COUNT_FRAME = 0
PERCLOS = 0.0
# loop over frames from the video stream
while True:
        
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	# loop over the face detections
	for (x, y, w, h) in rects:
            # construct a dlib rectangle object from the Haar cascade
            # bounding box
            rect = dlib.rectangle(int(x), int(y), int(x + w),
                    int(y + h))

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[bStart:bEnd]
            jaw = shape[qStart:qEnd]
            nose = shape[nStart:nEnd]
            
            
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mouthMAR = mouth_aspect_ratio(mouth)
            ang = distancia_nariz_queixo(nose,jaw)
            #print(ang)
            
            #print(mouth.lengt)
            #print(mouth)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            array.append(ear)
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            for o in range(0,6):
                x,y = leftEye[o]
                cv2.circle(frame,(x,y),1,(0,255,0),-1)
            for o in range(0,6):
                x,y = rightEye[o]
                cv2.circle(frame,(x,y),1,(0,255,0),-1)
            
            for o in range(0,len(mouth)):
                x,y = mouth[o]
                cv2.circle(frame,(x,y),1,(0,255,0),-1)
            
            x,y = nose[6]
            xj,yj = jaw[1]
            xjd, yjd = jaw[15]
            cv2.circle(frame,(x,y), 1, (0,255,0), -1)
            cv2.circle(frame,(xj,yj), 1, (0,255,0), -1)
            cv2.circle(frame,(xjd,yjd), 1, (0,255,0), -1)
            
            print(COUNT_FRAME)
            
            if COUNT_FRAME >= 100:
                COUNT_FRAME = 0
                n_olhos = len(array)
                ar = np.array(array)
                n_abertos = ar[ar>EYE_AR_THRESH]
                tam = len(n_abertos)
                print(n_olhos)
                print(n_abertos)
                print(COUNT_FRAME)
                array = []
                PERCLOS = ((n_olhos - tam)/n_olhos) * 100.0
                print(PERCLOS)
                
            
                #if PERCLOS > 80:
                #    cv2.putText(frame, "PERCLOS {:.2f}".format(PERCLOS), (250, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.putText(frame, "PERCLOS {:.2f}".format(PERCLOS), (250, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if ang < FACE_AR_THRESH:
                cv2.putText(frame, "Olhou para baixo", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if mouthMAR > MOUTH_AR_THRESH and flag_mouth == False:
                COUNTER_MOUTH_OPEN += 1
                flag_mouth = True
            if mouthMAR <= MOUTH_AR_THRESH:
                flag_mouth = False
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, "Fechado", (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if flag_eye == False:
                    COUNTER_EYE_CLOSE += 1
                    flag_eye = True
                
                # if the eyes were closed for a sufficient number of
                # frames, then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not ALARM_ON:
                                ALARM_ON = True

                                # check to see if the TrafficHat buzzer should
                                # be sounded
                                if args["alarm"] > 0:
                                        #th.buzzer.blink(0.1, 0.1, 10,background=True)
                                    print("alarmeeee")

                        # draw an alarm on the frame
                        cv2.putText(frame, "ACORDA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                flag_eye = False
                COUNTER = 0
                ALARM_ON = False
            COUNT_FRAME += 1
            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.3f}".format(mouthMAR), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MOP: {}".format(COUNTER_MOUTH_OPEN), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "CEC: {}".format(COUNTER_EYE_CLOSE), (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "HE: {:.3f}".format(ang), (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	# show the frame
	#COUNT_FRAME = 0
	
	cv2.imshow("Frame", frame)
	#cv2.waitKey(0)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	if key == ord("p"):
            cv2.imwrite("figura2.png",frame)
            print("gravou")
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
    
    
    
    
    
