import cv2
import os


def resize(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite("webwxgetmsgimg.jpg", resized)


if __name__ == '__main__':
    resize("webwxgetmsgimg.jpg")