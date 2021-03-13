#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-12-19 02:09:53
# @Author  : Gefu Tang (tanggefu@gmail.com)
# @Link    : https://github.com/primetang/pylsd
# @Version : 0.0.1

import cv2
import numpy as np
import os
from pylsd.lsd import lsd
from scipy import io






# MOT16-02-067
# MOT16-04-001
# MOT16-09-001
# MOT16-10-218
fullName = '/home/seungyeon/Desktop/git/neurvps/data/line/MOT16-02-067.jpg'
folder = '/home/seungyeon/Desktop/git/neurvps/data/line/'
name = 'MOT16-02-067'
#folder, imgName = os.path.split(fullName)

img = cv2.imread(fullName, cv2.IMREAD_COLOR)

imgLine = img.copy()
numLine = img.copy()


#############################################################[ find Line : LSD ]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
line = lsd(gray)
cnt = 0

lines = []
for i in range(line.shape[0]):
    pt1 = (int(line[i, 0]), int(line[i, 1]))
    pt2 = (int(line[i, 2]), int(line[i, 3]))
    width = line[i, 4]
    cv2.line(imgLine, pt1, pt2, (255, 0, 0), int(np.ceil(width / 2)))
    cv2.line(numLine, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
    lines.append(line[i][0:4])

    if choose_line and (cnt%10 == 0):
        #10번마다 번호저장
        cv2.putText(numLine, '%d' % i, pt1, cv2.FONT_HERSHEY_PLAIN, 1, color=(0, 0, 0))
    cnt += 1

#cv2.imwrite(os.path.join(folder, 'rst/lsd(py)_' + imgName.split('.')[0] + '.jpg'), imgLine)
cv2.imwrite(folder + 'rst/lsd(py)_' + name + '.jpg', imgLine)


if choose_line:
    # cv2.imshow('line numbers', numLine)
    if 'y' == input("저장 [y/n] ? n 추천 : "):
        cv2.imwrite(folder + 'rst/lsd(py)_num_' + name + '.jpg', numLine)




