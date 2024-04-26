import os
import cv2
import time
import glob

# print(cv2.__version__)
device = cv2.cuda.getCudaEnabledDeviceCount()
# print(device)


path = '/home/testimages'   # orginal image path
output = '/home/detection/yolov4_output'  # output path
curent_dir = os.path.dirname(__file__)
print('curent_dir', curent_dir)

for image in glob.glob(path + '/*.jpg'):
    print(image)
    # start = time.time()
    img = cv2.imread(image)

    with open(os.path.join(curent_dir, '/home/detection/obj.names'), 'r') as f:  # read the labels file
        classes = f.read().splitlines()

    net = cv2.dnn.readNetFromDarknet(os.path.join(curent_dir, '/home/detection/model_yolov4.cfg'),
                                     os.path.join(curent_dir, '/home/detection/weight/model_yolov4_last.weights'))

    model = cv2.dnn_DetectionModel(net)
    # set the input parameters image size and scale here
    model.setInputParams(scale=1 / 255, size=(736, 736), swapRB=True)

    # detect the objects threshold set here
    classIds, scores, boxes = model.detect(
        img, confThreshold=0.20, nmsThreshold=0.10)

    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)

        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=2)
        img_name = os.path.basename(image)
        cv2.imwrite(os.path.join(output, img_name), img)
    # end = time.time()
    # img_name= os.path.basename(image)
    # cv2.imwrite(os.path.join(output, img_name), img)
    # print('executatio:', end-start )

# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
