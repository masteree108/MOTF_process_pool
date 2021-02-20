# USAGE
# python MOTF_process_pool.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# watch CPU loading on the ubuntu
# ps axu | grep [M]OTF_process_pool.py | awk '{print $2}' | xargs -n1 -I{} ps -o sid= -p {} | xargs -n1 -I{} ps --forest -o user,pid,ppid,cpuid,%cpu,%mem,stat,start,time,command -g {}

# import the necessary packages
from imutils.video import FPS
from multiprocessing import Pool
import numpy as np
import argparse
import imutils
import cv2
import os

def get_algorithm_tracker(algorithm):
    if algorithm == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif algorithm == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif algorithm == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif algorithm == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif algorithm == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif algorithm == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif algorithm == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif algorithm == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    return tracker

def read_user_input_info():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
    ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    return args

def init_tracker(core_num, box, frame):
    print("core_num:%d" % core_num)
    # grab the appropiate object tracker using our dictionary of 
    # OpenCV object tracker objects
    # it should brings (left, top, width, height) to tracker.init() function
    # parameters are left, top , right and bottom in the box 
    # so those parameters need to minus like below to get width and height 
    bbox = (box[0], box[1], abs(box[0]-box[2]), abs(box[1]-box[3]))
    multi_tracker.add(get_algorithm_tracker("CSRT"), frame, bbox) 


def detect_people_and_tracker_init():
    # detecting how many person on this frame
    person_num = 0
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            #print("label:%s" % label)
            if CLASSES[idx] != "person":
                continue
                
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            bb = (startX, startY, endX, endY)
            print(bb)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            print("label:%s" % label)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            init_tracker(person_num, bb, frame)
            person_num = person_num + 1 
    print("person_num:%d" % person_num)
    return person_num

def main(frame):
    # loop over frames from the video file stream
    while True:
	# grab the next frame from the video file
        (grabbed, frame) = vs.read()

	    # check to see if we have reached the end of the video file
        if frame is None:
            break
        
        frame = imutils.resize(frame, width=800)
        ok, bboxes = multi_tracker.update(frame)
        if ok:                                         
            for i, newbox in enumerate(bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                (startX, startY) = p1
                cv2.putText(frame, "preson", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    args = read_user_input_info()
    
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the video stream and output video writer
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args["video"])
    
    # for saving tracker objects
    multi_tracker = cv2.MultiTracker_create()

    # detected flag
    detection_ok = False

    # if below variable set to True, this result will not show tracking bbox on the video
    # ,it will show number on the terminal
    print_number_test_not_tracker = False

    # step 1. grab the frame dimensions and convert the frame to a blob
    # step 2. detecting how many people on this frame
    # step 1:
    (grabbed, frame) = vs.read()
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
    net.setInput(blob)
    detections = net.forward()
    # step 2:
    detect_people_and_tracker_init()

    # start the frames per second throughput estimator
    fps = FPS().start()

    # tracking person on the video
    main(frame)

