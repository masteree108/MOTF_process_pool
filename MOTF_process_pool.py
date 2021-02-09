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
import dlib
import cv2
import os

tracker_list = []
def init_tracker(core_num, box,rgb):
    print("core_num:%d" % core_num)
    tracker_list.append(dlib.correlation_tracker())
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    tracker_list[core_num].start_track(rgb, rect)

# for pool testing 
def map_test(i):
    print(i)

def start_tracker(input_data):  
    n_rgb = 0
    n_tracker = 1 

    tl_num = input_data[n_tracker]
    print("start_tracker, track_list[%d]" % tl_num)
    tracker_list[tl_num].update(input_data[n_rgb])
    pos = tracker_list[tl_num].get_position()
    # unpack the position object
    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())
    bbox = (startX, startY, endX, endY)
    return bbox

# construct the argument parser and parse the arguments
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

# start the frames per second throughput estimator
fps = FPS().start()

detection_ok = False
print_number_test_not_tracker = False
core_num = 0

# grab the frame dimensions and convert the frame to a blob
(grabbed, frame) = vs.read()
frame = imutils.resize(frame, width=600)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
net.setInput(blob)
detections = net.forward()

if print_number_test_not_tracker == False:
    # detecting how many person on this frame
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
            init_tracker(core_num, bb, rgb);
            core_num = core_num + 1

    if core_num >= (os.cpu_count()-1):
        pool = Pool(os.cpu_count()-1)
    else:
        print("pool core_num:%d" % core_num)
        pool = Pool(core_num)
else:
    pool = Pool(11)
    detection_ok = True

# loop over frames from the video file stream
while True:
	# grab the next frame from the video file
    if detection_ok == True:
        (grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
        if frame is None:
            break

	# resize the frame for faster processing and then convert the
	# frame from BGR to RGB ordering (dlib needs RGB ordering)
        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        detection_ok = True

    input_data = []
    for i in range(core_num):
        input_data.append([])
        input_data[i].append(rgb)
        input_data[i].append(i)

     
    if print_number_test_not_tracker == True:
        #pool.map_async(map_test, [1,2,3,4,5,6,7,8,9,10,11])
        pool.map(map_test, [1,2,3,4,5,6,7,8,9,10,11])
    else:
        # can not use map_async,otherwise it will not wait all trackers to finish the job,
        # it will just executing print("before operating cv2") directly
        #pool_output = pool.map_async(start_tracker, input_data)     
        #pool.close()
        #pool.join()
        pool_output = pool.map(start_tracker, input_data)    

        print("before operating cv2")
        for i in range(len(pool_output)):
            #print(pool_output[i][0])
            #print(box)
            (startX, startY, endX, endY) = pool_output[i]
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
            cv2.putText(frame, "preson", (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    print("before imshow")
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
