from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if (detection[2][2] > 5) and (detection[2][3] > 5):
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
            cv2.circle(img, (int((xmin + xmax) / 2), int((ymin + ymax) / 2)), 10, (0, 0, 255), 0)
    return img


netMain = None
metaMain = None
altNames = None
# cifar_netMain = None
# cifar_metaMain = None
# cifar_altNames = None


def YOLO():

    global metaMain, netMain, altNames
    global cifar_metaMain, cifar_netMain, cifar_altNames
    # configPath = "./pycfgweightsdata/yolo-obj-cut512x224.cfg"
    # weightPath = "./pycfgweightsdata/yolo-obj-cut512x224_last.weights"
    # metaPath = "./pycfgweightsdata/obj_cut_512x224.data"
    configPath = "data/yolov3-tiny_obj_test.cfg"
    weightPath = "data/yolov3-tiny_obj_best.weights"
    metaPath = "data/obj.data"
    # cifar_configPath = "./cfg/cifar65.cfg"

    # cifar_weightPath = "./backup/cifar65_20000.weights"
    # cifar_metaPath = "./data/cifar65.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    # if cifar_netMain is None:
    #     cifar_netMain = darknet.load_net_custom(cifar_configPath.encode(
    #         "ascii"), cifar_weightPath.encode("ascii"), 0, 1)  # batch size = 1
    # if cifar_metaMain is None:
    #     cifar_metaMain = darknet.load_meta(cifar_metaPath.encode("ascii"))
    # if cifar_altNames is None:
    #     try:
    #         with open(cifar_metaPath) as metaFH:
    #             metaContents = metaFH.read()
    #             import re
    #             match = re.search("names *= *(.*)$", metaContents,
    #                               re.IGNORECASE | re.MULTILINE)
    #             if match:
    #                 result = match.group(1)
    #             else:
    #                 result = None
    #             try:
    #                 if os.path.exists(result):
    #                     with open(result) as namesFH:
    #                         namesList = namesFH.read().strip().split("\n")
    #                         cifar_altNames = [x.strip() for x in namesList]
    #             except TypeError:
    #                 pass
    #     except Exception:
    #         pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    out = cv2.VideoWriter(
        "hand_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    bbox_count = 1
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # cropped = frame_read[204:399, 308:791]  #截取视频右下部分图片,1280x720
        # cropped = frame_read[20:244, 250:762]  # 截取视频右下部分图片,1000x350
        #cropped = frame_read[200:616, 600:1624]  # 输入视频为1920*1080，截成1024x416
        cropped = frame_read
        frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        # count = 0
        # for detection in detections:
        #     bounds = detection[2]
        #     yExtent = int(bounds[3])
        #     xExtent = int(bounds[2])
        #     # Coordinates are around the center
        #     xCoord = int(bounds[0] - bounds[2] / 2)
        #     yCoord = int(bounds[1] - bounds[3] / 2)
        #     boundingBox = [
        #         [xCoord, yCoord],
        #         [xCoord, yCoord + yExtent],
        #         [xCoord + xExtent, yCoord + yExtent],
        #         [xCoord + xExtent, yCoord]
        #     ]


            # 生成cifar训练集，不用可以注释
            # start_x = int(xCoord * 1920 / 512)
            # end_x = int((xCoord + xExtent) * 1920 / 512)
            # start_y = int(yCoord * 1080 / 224)
            # end_y = int((yCoord + yExtent) * 1080 / 224)
            # start_x = xCoord * 2
            # end_x = (xCoord + xExtent) * 2
            # start_y = yCoord * 2
            # end_y = (yCoord + yExtent) * 2

            # # img = io.imread(darknet_image)
            # if (boundingBox[0][1] > 0) and (boundingBox[1][1] > 0) and (boundingBox[0][0] > 0) and (boundingBox[2][0] > 0)\
            #             and (xExtent > 5) and (yExtent > 5):  # 按尺寸大小、比例对bbox进行筛除
            #     cut = frame_resized[boundingBox[0][1]:boundingBox[1][1], boundingBox[0][0]:boundingBox[2][0]]
            #     # print(cut)
            #     cut = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
            #     # cv2.namedWindow("cut", cv2.WINDOW_NORMAL)
            #     # cv2.imshow("cut", cut)
            #     # cv2.waitKey(0)
            #     cv2.imwrite("./cut.jpg", cut)
            #     # origin_cut = cropped[start_y - 8:end_y + 8, start_x - 8:end_x + 8]
            #     # cv2.imwrite("E:\BaiduNetdiskDownload\session_20200306_070745\yolo_out\\" + str(bbox_count) + ".png", origin_cut)
            #     bbox_count = bbox_count + 1
            #     # cifar分类
            #     imagePath_class = "./cut.jpg"
            #     cifar_im = darknet.load_image(imagePath_class.encode('ascii'), 0, 0)
            #     classes = darknet.classify(cifar_netMain, cifar_metaMain, cifar_im)
            #     detection = list(detection)
            #     print(classes[0][0])
            #     print(classes[0][1])
            #     if classes[0][1] < 0.6:
            #         detection[0] = b'unknown'
            #     else:
            #         detection[0] = classes[0][0]
            #     detection[1] = classes[0][1]
            #     detection = tuple(detection)
            #     detections[count] = detection
            # else:
            #     detection = list(detection)
            #     detection[0] = b'unknown'
            #     detection = tuple(detection)
            #     detections[count] = detection
            # count = count + 1
            #         # detection[1] = classes[0][1]
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
