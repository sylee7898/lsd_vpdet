# LSD로 라인찾고 JLinkage로 그룹핑
# ROI 지정해주고 이거 내에 오면 계산

from scipy import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
from collections import Counter



# MOT16-02-067
# MOT16-04-001
# MOT16-09-001
# MOT16-10-218
#fullName = '/home/seungyeon/Desktop/git/neurvps/data/line/MOT16-02-067.jpg'
folder = '/home/seungyeon/Desktop/git/neurvps/data/line/'
name = 'MOT16-09-001'
#folder, imgName = os.path.split(fullName)

img = cv2.imread(folder+name+'.jpg', cv2.IMREAD_COLOR)

'''
MOT 02
B : 0,730 / img.shape[1],img.shape[0]
R : 0, 0 / 530,410
G : 0, 0 / img.shape[1],img.shape[0] > 그냥 roiLine[2] = label이 3인거

MOT 09
전체 다 
'''
roi =  [[0,730, img.shape[1],img.shape[0]],
        [0, 0, 530,410],
        [0, 0, img.shape[1],img.shape[0]]]

def get_crosspt(line1, line2):
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2
    if x12 == x11 or x22 == x21:
        return 0, 0
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    b1 = y12 - m1 * x12
    b2 = y22 - m2 * x22
    # print("m1 : ", m1, "m2 : ", m2)
    if m1 == m2:
        print('parallel')
        return 0, 0

    y = (b1 * m2 - b2 * m1) / (m2 - m1)
    x = (y - b1) / m1
    # x=(y-b2)/m2 도 x같나?

    y = img.shape[0] - y
    return x, y


# 이미 구한 line 파일 받아서 vp정하기
def candidate_vps(img):
    mat_file = io.loadmat(folder + 'line/' + name + '_line.mat')

    lines = mat_file['lines']
    labels = mat_file['labels']

    ######################

    vp = []
    roiLine = []


    for LINE in range(0, len(lines)):  # 라인 다 스캔해서 같은라벨중 roi에 들어온 라인 추출
        for i in range(0,3) :
            if labels[LINE] == (i+1) :
                line = lines[LINE]
                # roi내에 line의 첫번째 점이 포함되면 roiLine[i]에 추가
                if (line[0] > roi[i][0] and line[0] < roi[i][2] and line[1] > roi[i][1] and line[1] < roi[i][3] ) :
                    roiLine[i].append(line1)
    print(roiLine[0])
    print(roiLine[0][1])
    for i in range(0,3) :
        for LINE in range(0, len(roiLine[i]-1)) :
            for LINE2 in range(LINE+1, len(roiLine[i])) :
                x, y = get_crosspt(roiLine[i][LINE], roiLine[i][LINE2])
                vp[i].append(x,y)



    sum = []

    for i in range(0,3) :
        for l in range(0,len(roiLine[i])):
            sum[i][0] += vp[i][l][0]
            sum[i][1] += vp[i][l][1]


    VP1 = (sumx[0] / len(vp[0]), sumy[0] / len(vp[0]))
    VP2 = (sumx[1] / len(vp[1]), sumy[1] / len(vp[1]))
    VP3 = (sumx[2] / len(vp[2]), sumy[2] / len(vp[2]))
    print(VP1, VP2, VP3)

    plt.imshow(img)
    for i in range(len(vp[0])):
        plt.scatter(vp[0][i], vp[0][i], s=0.2, color='blue')
    for i in range(len(vp[1])):
        plt.scatter(vp[1][i], vp[1][i], s=0.2, color='red')
    for i in range(len(vp[2])):
        plt.scatter(vp[2][i], vp[2][i], s=0.2, color='green')

    plt.scatter(VP1[0], VP1[1], s=2, color='black')
    plt.scatter(VP2[0], VP2[1], s=2, color='black')
    plt.scatter(VP3[0], VP3[1], s=2, color='black')

    print(VP1)
    print(VP2)
    print(VP3)

    plt.show()
    # plt.savefig('./result/vpdet/'+name+'.png', img)


candidate_vps(img)


