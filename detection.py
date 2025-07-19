import time
import cv2 as cv

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
def detectFaces(img):
    blob=cv.dnn.blobFromImage(img,1,model_res,model_meanSubValues)
    model.setInput(blob)
    detections = model.forward()
    return detections

def markDetections(detections,frame,frameWidth,frameHeight):
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7]*[frameWidth,frameHeight,frameWidth,frameHeight]
            x1, y1, x2, y2 = box.astype("int")
            cv.rectangle(frame,(x1, y1),(x2, y2) ,(0,0,0),3)
            print(x1,x2,y1,y2)
            return frame[y1:y2,x1:x2]
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
        




def main():
    try:
        startFrames()
    except (NoFrameException , InterruptedException) as e:
        print(e)

if __name__=="__main__":
    main()