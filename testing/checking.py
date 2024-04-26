import os
import cv2
import time
import glob

print('Version of cv2', cv2.__version__)
device = cv2.cuda.getCudaEnabledDeviceCount()

path = '/home/train/abnormal'   # orginal image path
output = '/home/detection/output'  # output path
curent_dir = os.path.dirname(__file__)
print('curent_dir', curent_dir)

with open(os.path.join(curent_dir, 'obj.names'), 'r') as f:  # read the labels file
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet(os.path.join(curent_dir, '/home/detection/model_yolov4.cfg'), os.path.join(
    curent_dir, '/home/detection/weight/model_yolov4_last.weights'))  # read the model file

model = cv2.dnn_DetectionModel(net)
# set the input parameters image size and scale here
model.setInputParams(scale=1 / 255, size=(736, 736), swapRB=True)


for image in glob.glob(path+'/*.jpeg'):  # for all images in the path
    # print('image', image)
    start = time.time()
    img = cv2.imread(image)

    # detect the objects threshold set here
    classIds, scores, boxes = model.detect( img, confThreshold=0.0, nmsThreshold=0.4)
    if len(classIds) == 0:
        print('class ID', classIds)
        print('number of abnormality class detected', len(classIds))
        print('no object detected')
        img_name = os.path.basename(image)
        print('img_name', img_name)
        file = open('train_abnormal_normal.txt', 'a+')
        file.write('{}\n'.format(img_name))
        file.close()
    else:
        img_name = os.path.basename(image)
        print('img_name', img_name)
        print("\n ")
        file = open('train_abnormal_abnormal.txt', 'a+')
        file.write('{}\n'.format(img_name))
        file.close()
