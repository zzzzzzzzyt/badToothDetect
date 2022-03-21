import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt
dirpath = 'toothDeep'
newdir = 'fill_toothDeep'

if not os.path.exists(newdir):
    os.makedirs(newdir)

order = [int(i.strip(".png")) for i in os.listdir(dirpath) if i.endswith(".png")]
jpglist = [f"{i}.png" for i in sorted(order)]  # 直接读取可能非顺序帧

for i, png in enumerate(jpglist):
    old = dirpath + f'/{png}'
    img = cv.imread(old)  # 返回的是numpy.array对象  返回的是数组 可能就不能用于下面的获取阈值



    # 将图像转换为
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # 数组的二值化

    # 自己设定阈值
    imgray[imgray < 100] = 0
    imgray[imgray >= 100] = 255

    # 根据算法获取阈值

    # 阈值取自相邻区域的平均值
    # imgray = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    # 阈值取值相邻区域的加权和，权重为一个高斯窗口。
    # imgray = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


    # 孔洞填充下面是

    # 原图取补得到MASK图像
    mask = 255 - imgray

    # 构造Marker图像
    marker = np.zeros_like(imgray)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255
    marker_0 = marker.copy()

    # 形态学重建  ksize是传递的内核大小
    # 内核形状矩形
    # SE = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3, 3))
    # 内核形状交叉型
    SE = cv.getStructuringElement(shape=cv.MORPH_CROSS, ksize=(3, 3))
    # 内核形状椭圆
    # SE = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3, 3))
    while True:
        marker_pre = marker
        dilation = cv.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    filling = dst - imgray

    # 进行反色处理
    # filling = 255 - filling
    # 形态学内核
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 先闭运算 再开
    # filling = cv.morphologyEx(filling, cv.MORPH_CLOSE, kernel, 1)
    # 形态学处理:开运算
    filling = cv.morphologyEx(filling, cv.MORPH_OPEN, kernel, 1)
    #

    # 将获得二值化 的轮廓画上去
    # contours, hierarchy = cv.findContours(filling, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 只要最外轮廓
    contours, hierarchy = cv.findContours(filling, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 选取最大轮廓进行填充
    area = []
    for k in range(len(contours)):
        area.append(cv.contourArea(contours[k]))
    if area!=[]:
        max_idx = np.argmax(np.array(area))



    # 最后个是线条厚度
    cv.drawContours(img, contours, max_idx, (212, 255, 127), 1)

    # 显示
    # plt.figure(figsize=(12, 6))  # width * height
    # plt.subplot(2, 3, 1), plt.imshow(imgray, cmap='gray'), plt.title('src'), plt.axis("off")
    # plt.subplot(2, 3, 2), plt.imshow(mask, cmap='gray'), plt.title('Mask'), plt.axis("off")
    # # plt.subplot(2, 3, 3), plt.imshow(marker_0, cmap='gray'), plt.title('Marker 0'), plt.axis("off")
    # plt.subplot(2, 3, 3), plt.imshow(img, cmap='gray'), plt.title('Marker 0'), plt.axis("off")
    # plt.subplot(2, 3, 4), plt.imshow(marker, cmap='gray'), plt.title('Marker'), plt.axis("off")
    # plt.subplot(2, 3, 5), plt.imshow(dst, cmap='gray'), plt.title('dst'), plt.axis("off")
    # plt.subplot(2, 3, 6), plt.imshow(filling, cmap='gray'), plt.title('Holes'), plt.axis("off")



    new = newdir + f'/{png}'
    # cv2.imwrite(new, invert)
    cv.imwrite(new, img)
    print(f'{i + 1} / {len(jpglist)}')
