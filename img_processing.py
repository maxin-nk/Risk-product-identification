# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin
import io, os
import numpy as np
import cv2
import re
from PIL import Image, ImageEnhance, ImageChops


class PicExpansion:

    # =====================================图像扩增
    def __init__(self, path):
        """
        初始化图片文件路径
        :param path: 初始路径
        """
        self.path = path  # 起始路径
        self.sub = []  # 子文件夹下文件的个数
        self.num = 0  # 文件总数

    def get_Imgs_Paths(self, *suffix):
        """
        批量获取图片名称
        :return:文件夹下的所有图像名称列表
        """
        trees = os.walk(self.path)

        pathArray = []
        # 获取所有文件的绝对路径
        # [str1, str2, ...]
        for root, dirs, files in trees:
            self.sub.append(len(files))
            for fn in files:
                if os.path.splitext(fn)[1] in suffix:  # 判断图片文件的格式
                    fname = os.path.join(root, fn)
                    pathArray.append(fname)
        self.num = sum(self.sub)
        return pathArray

    def reSize(self, filepaths, *suffix):
        """
        原始图像大小调整
        :param filepaths:
        :return:
        """
        for i in range(self.num):
            img = Image.open(filepaths[i])
            img2 = img.resize(suffix)
            img2.save(filepaths[i])
        return None

    def move(self, filepath):
        """
        图像平移
        :return:
        """
        img = Image.open(filepath)
        tmp = min(img.height, img.width) / 10
        return ImageChops.offset(img, np.random.randint(0, tmp, 1))

    def flip(self, filepath):
        """
        图像翻转
        :return:
        """
        img = Image.open(filepath)
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def rotation(self, filepath):
        """
        随机角度旋转
        :param filepaths:
        :return:
        """
        img = Image.open(filepath)
        return img.rotate(angle=np.random.randint(0, 360, 1))

    def changeColor(self, filepath):
        """
        随机颜色
        :param filepaths:
        :return:
        """
        img = cv2.imread(filepath)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + np.random.randint(1, 180)) % 180
        img_hsv[:, :, 1] = (img_hsv[:, :, 1] + np.random.randint(1, 255)) % 255
        img_hsv[:, :, 2] = (img_hsv[:, :, 2] + np.random.randint(1, 255)) % 255
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def changeBright_and_contrast(self, filepath, alpha=1, beta=0):
        """
        调整图像亮度 + 对比图
        :param filepath:
        :param alpha: 对比度参数
         :param beta: 亮度参数
        :return:
        """
        img = Image.open(filepath)
        return Image.eval(img, lambda x: x * alpha + beta)

    # def deNoisingColored(self, filepath):
    #     """
    #     去除彩色图像高斯噪声
    #     :param filepath:
    #     :return:
    #     """
    #     img = cv2.imread(filepath)
    #     return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    #
    # def local_threshold(self, filepath):
    #     """
    #     灰度图二值化（局部阙值）
    #     :param filepath:
    #     :return:
    #     """
    #     img = cv2.imread(filepath)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     ret, binary = cv2.threshold(img, 2*(np.max(img)-np.min(img))/5, 255, cv2.THRESH_BINARY)
    #     return binary

    # =====================================图像预处理
    def box_detect(self, filepath):
        """
        边缘检测及裁剪
        :param filepath:
        :return:
        """
        # 转换为灰度图
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = Image.open(filepath)

        # 留下具有高水平梯度和低垂直梯度的图像区域
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.addWeighted(gradX, 1, gradY, 2, 1)
        # gradient = cv2.subtract(gradX, gradY)
        # Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
        # 因此要使用16位有符号的数据类型，即cv2.CV_16S。处理完图像后，再使用cv2.convertScaleAbs()函数将其转回原来的uint8格式，
        # 否则图像无法显示。
        gradient = cv2.convertScaleAbs(gradient)
        cv2.imwrite(str(np.random.randint(100, 200, 1)) + ".jpg", gradient)

        # 均值滤波
        blurred = cv2.blur(gradient, (9, 9))
        # ret, binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
        # ret, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
        # 二值化
        (ret, thresh) = cv2.threshold(blurred, 2 * (np.max(img) - np.min(img)) / 5, 255, cv2.THRESH_BINARY)

        # 图像滤波
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))  # 返回指定形状及尺寸的结构元素
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 形态学滤波

        # 腐蚀与膨胀
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
        closed = cv2.erode(closed, element, iterations=4)
        closed = cv2.dilate(closed, element, iterations=4)
        # cv2.imwrite(str(np.random.randint(100, 200, 1))+".jpg", closed)

        # 边缘检测
        cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cc = cv2.Canny(closed.copy(), 80, 150)
        ccc = cv2.Canny(closed.copy(), 50, 100)
        ret = np.hstack((cc, ccc))
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)  # 得到最小外接旋转矩形的（中心(x,y), (宽,高), 旋转角度）
        box = np.int0(cv2.boxPoints(rect))  # cv2.boxPoints:得到最小外接旋转矩形的四个顶点。用于绘制旋转矩形。

        # 裁剪
        Xs = [i[0] for i in box]  # 获取4个顶点x坐标
        Ys = [i[1] for i in box]  # 获取4个顶点y坐标
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1

        # # 给原图画线
        # out = cv2.rectangle(img, (x1, y1), (x1 + width, y1 + hight), (0, 0, 255), 20)
        # cv2.imwrite(str(np.random.randint(100, 200, 1)) + ".jpg", out)

        return cv2.resize(img[y1:y1 + hight, x1:x1 + width], (32, 32))


if "__main__" == __name__:
    path = "E:\\picOriginal"
    test = PicExpansion(path)
    filepaths = test.get_Imgs_Paths(".jpg", ".png")

    # 图像增强方法的个数
    fuc = 2

    # 操作循环i次
    for i in range(1):
        # 分文件夹操作
        for j in range(1, len(test.sub)):
            tmp = 0
            # 遍历商品路径列表
            for k in range(test.num):
                cls = int(filepaths[k].split("\\")[-2])
                if cls == j:
                    # tmp += 1
                    # img = test.move(filepaths[k])  # 图像平移
                    # img.save(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg")
                    # tmp += 1
                    # img = test.flip(filepaths[k])  # 图像翻转
                    # img.save(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg")
                    # tmp += 1
                    # img = test.rotation(filepaths[k])   # 随机角度旋转
                    # img.save(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg")
                    # tmp += 1
                    # img = test.changeColor(filepaths[k])  # 颜色变换
                    # cv2.imwrite(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg", img)
                    # tmp += 1
                    # img = test.changeBright_and_contrast(filepaths[k], 1, beta=80)   # 亮度调节
                    # img.save(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg")
                    # tmp += 1
                    # img = test.changeBright_and_contrast(filepaths[k], 4, 0)   # 对比度调节
                    # img.save(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg")
                    # tmp += 1
                    # img = test.cvtColor_gray(filepaths[k])  # 灰度图转化
                    # cv2.imwrite(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg", img)
                    # tmp += 1
                    # img = test.deNoisingColored(filepaths[k])  # 去除图像噪声
                    # cv2.imwrite(filepaths[k].split("_")[0]+"_"+str(test.sub[j]+tmp)+".jpg", img)
                    # tmp += 1
                    # img = test.local_threshold(filepaths[k])
                    # cv2.imwrite(filepaths[k].split("_")[0] + "_" + str(test.sub[j] + tmp) + ".jpg", img)
                    tmp += 1
                    img = test.box_detect(filepaths[k])
                    # cv2.imwrite(filepaths[k].split("_")[0] + "_" + str(test.sub[j] + tmp) + ".jpg", img)

            test.sub[j] = test.sub[j] * (fuc + 1)