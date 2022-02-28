from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
# we defiine the object detection class

detector = ObjectDetection()  

#set the model type to RetinaNet 
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
#load the model into the object detection class
detector.loadModel()

# we called the detection function and parsed in the input image path and the output image path 

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "github.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )