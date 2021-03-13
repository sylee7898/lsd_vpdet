#################################################
# LSD로 라인찾고 3그룹 따로 라인 두개씩 선택해 vp 구함 ##
#################################################


import cv2
import numpy as np
import os
from LSD.pylsd.pylsd.lsd import lsd
from scipy import io
import random





vpnum = int(input("vpnum [B 1, R 2, G 3] : "))
if 'y' == input("choose the line [y/n] : "):
    choose_line = True
else:
    choose_line = False

#choose_line = False



def get_crosspt(line1, line2):
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2


    if x12==x11 or x22==x21:
        return 0
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    #print("m1 : ", m1, "m2 : ", m2)
    if m1==m2:
        print('parallel')
        return 0
    #print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)

    # img 중심에서 원점 대칭

    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    # y축을 영상 중심으로부터 뒤집기
    #cy = img.shape[0] - cy
    #cx = img.shape[1] - cx

    return cx, cy



# MOT16-02-067
# MOT16-04-001
# MOT16-09-001
#fullName = '/home/seungyeon/Desktop/git/neurvps/data/line/MOT16-02-067.jpg'
folder = '/home/seungyeon/Desktop/git/neurvps/data/line/'
name = '1'
# 0, 1, 6, 11, 17, 57, 59
#folder, imgName = os.path.split(fullName)

img = cv2.imread(folder+name+'.jpeg', cv2.IMREAD_COLOR)

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

    # J-linkage로 라벨 뽑아 vpnum과 라벨 같은것만 숫자표기 어떰 ??????????????????????
    if choose_line :
        #if (cnt%5 == 0) :
            # 5번마다 번호저장
            cv2.putText(numLine, '%d' % i, pt1, cv2.FONT_HERSHEY_PLAIN, 1, color=(random.random(), random.random(), random.random()))
    cnt += 1

#cv2.imwrite(os.path.join(folder, 'rst/lsd(py)_' + imgName.split('.')[0] + '.jpg'), imgLine)
cv2.imwrite(folder + 'rst/lsd(py)_' + name + '.jpeg', imgLine)


if choose_line:
    # cv2.imshow('line numbers', numLine)
    if 'y' == input("저장 [y/n] ? n 추천 : "):
        cv2.imwrite(folder + 'rst/lsd(py)_num_' + name + '.jpeg', numLine)

###########################################################################[ find vp : choose line ]
h,  w = img.shape[:2]        # 576, 704
cx = w / 2
cy = h / 2


grid_size = 70



if choose_line:


    # 두 수직선 골라 중간 horizontal line 구하기
    horizontal1 = int(input("1 line number : "))  # 왼쪽
    horizontal2 = int(input("2 line number : "))  # 오른쪽
    print(lines[horizontal1])
    print(lines[horizontal2])

    # vanishing point 구하기
    vanPx, vanPy = get_crosspt(lines[horizontal1], lines[horizontal2])

    if vpnum == 1 :
        #vanPy += 200

        print("vp : ", vanPx, vanPy)
        vanishing = img.copy()
        '''
        # 선택한 라인 노란색으로 표시
        cv2.line(vanishing, (int(lines[horizontal1][0]), int(lines[horizontal1][1])),
                 (int(lines[horizontal1][2]), int(lines[horizontal1][3])), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(vanishing, (int(lines[horizontal2][0]), int(lines[horizontal2][1])),
                 (int(lines[horizontal2][2]), int(lines[horizontal2][3])), (0, 255, 255), 2, cv2.LINE_AA)
        '''

        color = [255, 0, 0]     # B
        # horizon 가로
        '''
        for i in range(-9, 10):
            # 9개의 점 vanishing point랑 이을거야
            CY = cy + i * grid_size
            slop = -(CY - vanPy) / (cx - vanPx)
            t = CY - slop * cx
            # y = 0일때 x = -t / slop
            x1 = int(-t / slop)  # y = 0인 지점 x1
            if i == 0:
                xx = x1

            # y = h 일때 x = (h - t) / slop
            x2 = int((h - t) / slop)

            cv2.line(vanishing, (x1, 0), (x2, h), (0, 0, 255), 1, cv2.LINE_AA)  # horizon
        cv2.imwrite(folder + 'rst/lsd(py)_vanishingLine1_' + name + '.jpg', vanishing)
        '''



    elif vpnum == 2 :
        #vanPy -= 50
        print("vp : ", vanPx, vanPy)
        vanishing = cv2.imread(
            '/home/seungyeon/Desktop/git/neurvps/data/line/rst/lsd(py)_vanishingLine1_'+ name +'.jpg', cv2.IMREAD_COLOR)

        '''
        cv2.line(vanishing, (int(lines[horizontal1][0]), int(lines[horizontal1][1])), (int(lines[horizontal1][2]), int(lines[horizontal1][3])), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(vanishing, (int(lines[horizontal2][0]), int(lines[horizontal2][1])), (int(lines[horizontal2][2]), int(lines[horizontal2][3])), (0, 255, 255), 2, cv2.LINE_AA)
        '''
        color = [0, 0, 255]     # R
        # cx,cy말고 다른 점들도 vanishing poin랑 이어주기
        # horizon 세로
        '''
        for i in range(-9, 10):
            # 9개의 점 vanishing point랑 이을거야
            CX = cx + i * grid_size
            slop = -(cy - vanPy) / (CX - vanPx)
            t = cy - slop * CX
            # y = 0일때 x = -t / slop
            x1 = int(-t / slop)  # y = 0인 지점 x1
            if i == 0:
                xx = x1

            # y = h 일때 x = (h - t) / slop
            x2 = int((h - t) / slop)

            cv2.line(vanishing, (x1, 0), (x2, h), (255, 0, 0), 1, cv2.LINE_AA)  # horizon
        cv2.imwrite(folder + 'rst/lsd(py)_vanishingLine2_' + name + '.jpg', vanishing)
        '''

    else : # vpnum == 3
        print("vp : ", vanPx, vanPy)
        vanishing = cv2.imread(
            '/home/seungyeon/Desktop/git/neurvps/data/line/rst/lsd(py)_vanishingLine2_'+ name +'.jpg', cv2.IMREAD_COLOR)
        '''
        cv2.line(vanishing, (int(lines[horizontal1][0]), int(lines[horizontal1][1])),
                 (int(lines[horizontal1][2]), int(lines[horizontal1][3])), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(vanishing, (int(lines[horizontal2][0]), int(lines[horizontal2][1])),
                 (int(lines[horizontal2][2]), int(lines[horizontal2][3])), (0, 255, 255), 2, cv2.LINE_AA)
        '''
        color = [0, 255, 0]     # G




    # 아래
    for i in range(0,img.shape[1],int(img.shape[1]/10)):
        cv2.line(vanishing, (int(vanPx), int(vanPy)), (i, img.shape[0]), color, 1, cv2.LINE_AA)
    cv2.line(vanishing, (int(vanPx), int(vanPy)), (img.shape[1], img.shape[0]), color, 1, cv2.LINE_AA)

    # 오른쪽
    for i in range(0, img.shape[0], int(img.shape[0] / 10)):
        cv2.line(vanishing, (int(vanPx), int(vanPy)), (img.shape[1], i), color, 1, cv2.LINE_AA)
    cv2.line(vanishing, (int(vanPx), int(vanPy)), (img.shape[1], img.shape[0]), color, 1, cv2.LINE_AA)

    # 왼쪽
    for i in range(0, img.shape[0], int(img.shape[0] / 10)):
        cv2.line(vanishing, (int(vanPx), int(vanPy)), (0, i), color, 1, cv2.LINE_AA)
    cv2.line(vanishing, (int(vanPx), int(vanPy)), (0, img.shape[0]), color, 1, cv2.LINE_AA)

    # 위
    for i in range(0, img.shape[1], int(img.shape[1] / 10)):
        cv2.line(vanishing, (int(vanPx), int(vanPy)), (i, 0), color, 1, cv2.LINE_AA)
    cv2.line(vanishing, (int(vanPx), int(vanPy)), (img.shape[1], 0), color, 1, cv2.LINE_AA)

    cv2.imwrite(folder + 'rst/lsd(py)_vanishingLine'+ (str)(vpnum) +'_'+ name +'.jpg', vanishing)


    # plot에 그리기
    '''
    vp0 = []
    vp1 = []
    for i in range(0,3) :
        color = (random.random(), random.random(), random.random())
        plt.scatter(vp0, vp1, color=color, s=1)

        ##### 이거부터 for문 추가해서 그림
        for xy in np.linspace(0, 512, 10):  # 이미지 상하좌우 4면에 10개씩 줄긋기  (4개씩 10번)
            line = plt.plot(
                # [vp[1], xy, vp[1], xy, vp[1], 0, vp[1], 511],
                # [vp[0], 0, vp[0], 511, vp[0], xy, vp[0], xy],
                [vp0, xy, vp0, xy, vp0, 0, vp0, 511],
                [vp1, 0, vp1, 511, vp1, xy, vp1, xy],
            )
            plt.setp(line, color=color, linewidth=0.1)
    '''
    




cv2.waitKey()




