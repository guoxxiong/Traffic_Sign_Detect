import cv2
import os

base = "./classifier_train/"
image_count = 57216

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname, f


for i, name in findAllFile(base):
    if i.endswith('.png'):
        image = cv2.imread(i)
        image_shape = image.shape
        if (image_shape[0] * image_shape[1]) > 10000:
            img1 = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(image, (0, 0), fx=0.35, fy=0.35, interpolation=cv2.INTER_CUBIC)
            img3 = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            img4 = cv2.resize(image, (0, 0), fx=0.22, fy=0.22, interpolation=cv2.INTER_CUBIC)
            img5 = cv2.resize(image, (0, 0), fx=0.20, fy=0.20, interpolation=cv2.INTER_CUBIC)
            img6 = cv2.resize(image, (0, 0), fx=0.18, fy=0.18, interpolation=cv2.INTER_CUBIC)
            img7 = cv2.resize(image, (0, 0), fx=0.16, fy=0.16, interpolation=cv2.INTER_CUBIC)
            img8 = cv2.resize(image, (0, 0), fx=0.14, fy=0.14, interpolation=cv2.INTER_CUBIC)
            img9 = cv2.resize(image, (0, 0), fx=0.13, fy=0.13, interpolation=cv2.INTER_CUBIC)
            img10 = cv2.resize(image, (0, 0), fx=0.12, fy=0.12, interpolation=cv2.INTER_CUBIC)
            img11 = cv2.resize(image, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
            img12 = cv2.resize(image, (0, 0), fx=0.08, fy=0.08, interpolation=cv2.INTER_CUBIC)
            name = list(name)
            #for name_count in range(len(name)):
                #name.pop(0)
                #if name[0] == '_':
                    #break
            name = ''.join(name)
            cv2.imwrite('./generated/' + str(image_count) + "_" +name, img1)
            cv2.imwrite('./generated/' + str(image_count + 1) + "_" + name, img2)
            cv2.imwrite('./generated/' + str(image_count + 2) + "_" + name, img3)
            cv2.imwrite('./generated/' + str(image_count + 3) + "_" + name, img4)
            cv2.imwrite('./generated/' + str(image_count + 4) + "_" + name, img5)
            cv2.imwrite('./generated/' + str(image_count + 5) + "_" + name, img6)
            cv2.imwrite('./generated/' + str(image_count + 6) + "_" + name, img7)
            cv2.imwrite('./generated/' + str(image_count + 7) + "_" + name, img8)
            cv2.imwrite('./generated/' + str(image_count + 8) + "_" + name, img9)
            cv2.imwrite('./generated/' + str(image_count + 9) + "_" + name, img10)
            cv2.imwrite('./generated/' + str(image_count + 10) + "_" + name, img11)
            cv2.imwrite('./generated/' + str(image_count + 11) + "_" + name, img12)
            image_count = image_count + 12
