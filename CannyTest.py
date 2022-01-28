import os

import cv2
import numpy as np

dirpath = 'toothLines'
newdir = 'Lines_tooth'

if not os.path.exists(newdir):
    os.makedirs(newdir)

order = [int(i.strip(".png")) for i in os.listdir(dirpath) if i.endswith(".png")]
jpglist = [f"{i}.png" for i in sorted(order)]  # 直接读取可能非顺序帧

for i, png in enumerate(jpglist):
    old = dirpath + f'/{png}'
    img = cv2.imread(old)  # 返回的是numpy.array对象

    # 灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 降噪
    gray = cv2.fastNlMeansDenoising(gray)
    # 二值化
    ret, invert = cv2.threshold(~gray, 190, 255, cv2.THRESH_BINARY)

    # 提取边缘算子
    # edge = cv2.Canny(invert, 30, 255)
    # invert = np.abs((255 * np.ones((edge.shape)) - edge)).clip(0, 255)

    # 找到轮廓
    contours,h = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # drawContours 轮廓
    sure = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # cv2.imshow('img', invert)
    new = newdir + f'/{png}'
    # cv2.imwrite(new, invert)
    cv2.imwrite(new, sure)
    print(f'{i + 1} / {len(jpglist)}')
