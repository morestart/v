import cv2
import os


# print(os.listdir("/home/ctl/PycharmProjects/ML/data/train/cat"))
name = os.listdir("data/train/cat")
# print(len(name))
# print(name[1818])
for i in range(len(name)):
    # print("/home/ctl/PycharmProjects/ML/data/train/cat/" + name[i])
    img = cv2.imread("data/train/cat/" + name[i])

    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite("data/train/cat/" + name[i], resized)
    print(name[i] + " resized" + "第" + str(i) + "张")

