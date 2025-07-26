import time
import cv2 as cv
from deepface import DeepFace
import ctypes

#TypeError: recognize() missing 1 required positional argument: 'frame'

#vars
face = "image.png"

#dnn detection model init
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
model = cv.dnn.readNetFromCaffe(configFile,modelFile)
model_res = (300, 300)
model_meanSubValues = (104, 177, 123)

#CV video init
cam = cv.VideoCapture(0)

#exceptions
class NoFrameException(Exception):
    def __init__(self, *args):
        super().__init__(*args)
        self.message = "No frame was captured from the video source. \nPlease check if the camera is connected, accessible, or the video file path is correct."
    def __str__(self):
        return f'\033[91m{self.message}\033[0m'
class InterruptedException(Exception):
    def __init__(self, *args):
        super().__init__(*args)
        self.message = "The operation was interrupted by clicking waitkey."
    def __str__(self):
        return f'\033[91m{self.message}\033[0m'


#functions
def storeFace():
    OK , frame = cam.read()

    if not OK:
        raise NoFrameException
        
    if cv.waitKey(1) & 0xFF == ord('m'):
        raise InterruptedException
    
    height , width = frame.shape[0:2]
    
    detections = detectFaces(frame)

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7]*[width,height,width,height]
            x1, y1, x2, y2 = box.astype("int")
            x1, y1, x2, y2 = x1-50, y1-50, x2+50, y2+50
            cropped=frame[x1:x2,y1:y2]
        cv.imwrite("image.png",cropped)

def detectFaces(img):
    blob=cv.dnn.blobFromImage(img,1,model_res,model_meanSubValues)
    model.setInput(blob)
    detections = model.forward()
    return detections

def markDetections(detections,frame,frameWidth,frameHeight):
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.3:
            box = detections[0,0,i,3:7]*[frameWidth,frameHeight,frameWidth,frameHeight]
            x1, y1, x2, y2 = box.astype("int")
            x1, y1, x2, y2 = x1-20, y1-20, x2+20, y2+20
            cv.rectangle(frame,(x1, y1),(x2, y2) ,(0,0,0),3)
            cropped=frame[x1:x2,y1:y2]
            verification = DeepFace.verify(img1_path = "image.png", img2_path = cropped, enforce_detection = False , model_name = "ArcFace",detector_backend='mtcnn')
            distance = verification['distance']
            threshold = 0.68

            if distance < threshold:
                print(True)
            else:
                print(False)
                
            return frame
        print("no face")
        return frame
        
def startFrames():
    while True:
        OK , frame = cam.read()

        if not OK:
            raise NoFrameException
        
        if cv.waitKey(1) & 0xFF == ord('m'):
            raise InterruptedException
        
        
        
        height , width = frame.shape[0:2]

        detections = detectFaces(frame)

        frame = markDetections(detections=detections,frame=frame,frameWidth=width,frameHeight=height)

        
        cv.imshow("frames",frame)

        time.sleep(0.0167)
        

def main():
    try:
        startFrames()
    except (NoFrameException , InterruptedException) as e:
        print(e)

def main2():
    storeFace()

if __name__=="__main__":
    main()