import cv2 as cv

#dnn detection model init
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
model = cv.dnn.readNetFromCaffe(configFile,modelFile)
model_res = (300, 300)
model_meanSubValues = (104, 177, 123)

#CV video init
capture = cv.VideoCapture(0)

def detection(img):
    blob=cv.dnn.blobFromImage(img,1,model_res,model_meanSubValues)
    model.setInput(blob)
    detections = model.forward()

    return detections





def main():
    pass


if __name__=="__main__":
    main()