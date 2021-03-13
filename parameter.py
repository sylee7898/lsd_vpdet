# vp 3점 x,y,z 있을 때
# parameter 구하는 법

import os
import io
import glob
import cv2
import numpy as np
import math

import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import json

#rootdir = './data/test/frame-000000-'
#rootdir = './logs/200331-103334-967be13-tmm17/npz/8000/'
#rootdir = './data/su3_/000/'
#rootdir = './data/scannet-vp/scene0055_02/'
rootdir = './data/my/'



# MOT16-02-067
# MOT16-09-001
# camera1_bg
# camera3_bg
name = 'camera1_bg'     #32
#name = 'frame-000670'
img = cv2.imread(rootdir + name +'.jpg')
#img = cv2.imread(rootdir + name +'-color.png')
#img = cv2.imread('/home/seungyeon/Desktop/git/neurvps/data/line/camera1_bg.jpg', cv2.IMREAD_COLOR)

f = 2.185     #2.4
cx = img.shape[0]/2
cy = img.shape[1]/2


# SU3 npz : vpts
'''
with open(rootdir + name +"_camera.json") as file:
    js = json.load(file)
    RT = np.array(js["modelview_matrix"])
'''
with np.load(rootdir + name +"_label.npz") as npz:
    vpts = np.array(npz["vpts"])      # vpts = (3,3) : [vp1: x,y,z] [vp2: x,y,z] [vp3: x,y,z]
    print(vpts)
    #print(vpts[0][0] / vpts[0][2] * f * 256 + cx, -vpts[0][1] / vpts[0][2] * f * 256 + cy)
    #print(vpts[1][0] / vpts[1][2] * f * 256 + cx, -vpts[1][1] / vpts[1][2] * f * 256 + cy)
    #print(vpts[2][0] / vpts[2][2] * f * 256 + cx, -vpts[2][1] / vpts[2][2] * f * 256 + cy)
    vpts = []

    for axis in [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]:
        vp = RT @ axis
        vp = np.array([vp[0], vp[1], -vp[2]])  # z 부호 반대로 > z축은 위로 향해야해서?
        vp /= np.linalg.norm(vp)
        vpts.append(vp)  # RT중 R인 3*3부분 T해줌 > [x,y,z],[x,y,z],[x,y,z] = v0
        # [0,1]로 계산한거라 256 곱하고 // 영상 중심점 (0,0)으로 잡은거라 픽셀좌표 절반으로 옮겨
    print("vpts : ", vpts)


'''
vp0 = (vp[0] / vp[2] * f * 256 + cx)    # x/z  * f * 256 + 256
vp1 = (-vp[1] / vp[2] * f * 256 + cy)
vp0,vp1은 vpts를 이미지상의 2D 점으로 매칭시킨거다
이걸로 R 구하면 ([1,0,0],[0,0,-1],[0,1,0]) 나옴
'''



#with np.load(rootdir + name +"-vanish.npz") as npz:
#    vpts = np.array([npz[d] for d in ["x", "y", "z"]])
#    print(vpts)
    # vpts는 3 by 3

'''
vpts = [x,y,z], [x,y,z], [x,y,z]

########## su3외에는 이거 필요 ##########################
vp0 = []
vp1 = []
vp0.append(vp[0] / vp[2] * f * 256 + cx)
vp1.append(-vp[1] / vp[2] * f * 256 + cy)

vp0 = [vpts[0][0]/vpts[0][2], vpts[1][0]/vpts[1][2], vpts[2][0]/vpts[2][2]]    # x
vp1 = [vpts[0][1]/vpts[0][2], vpts[1][1]/vpts[1][2], vpts[2][1]/vpts[2][2]]    # y
'''
###### npz 만들기 #################################됨
'''
MOT02=[[-4863, 630, 1], [727, 467, 1], [1206, 23776, 1]]
MOT09=[[204, -16510, 1], [-516.88, 656.77, 1], [2565.69, 495.98, 1]]
cam3=[[-9669.93, -644.53, 1], [1171, -615, 1], [16.45, 4446, 1]]
cam1=[[5032, -92, 1], [-541, -916, 1], [1229, 422252, 1]]

np.savez('./data/test/MOT16-02_label', MOT02)
np.savez('./data/test/MOT16-09_label', MOT09)
np.savez('./data/test/camera1_bg_label', cam1)
np.savez('./data/test/camrea3_bg_label', cam3)      # npz안에 npy 이름 arr_0으로 저장
'''

##### new f 찾기 ###########################
#### f구하기 :  vp & calib 논문 ############################################################
'''
x1x2 + y1y2 + ff = 0
x2x3 + y2y3 + ff = 0
x3x1 + y3y1 + ff = 0
'''
# npz에 저장된 vpt는 world 좌표
# 이미지 2D좌표로 변환하려면 f*256+cx 곱해줘야해 >> f 알아야해
'''
x1 = vpts[0][0] / vpts[0][2]#  * f * 256 + cx
y1 = vpts[0][1] / vpts[0][2]#  * f * 256 + cy
x2 = vpts[1][0] / vpts[1][2]#  * f * 256 + cx
y2 = vpts[1][1] / vpts[1][2]#  * f * 256 + cy
x3 = vpts[2][0] / vpts[2][2]#  * f * 256 + cx
y3 = vpts[2][1] / vpts[2][2]#  * f * 256 + cy
'''
'''
# MOT02
x1 = -4863
y1 = 630
x2 = 727
y2 = 467
x3 = 1209
y3 = 23776

# MOT09
x1 = 204
y1 = -16510
x2 = -516
y2 = 656
x3 = 2565
y3 = 495

#cam1
x1 = 5032
y1 = -92
x2 = -541
y2 = -916
x3 = 1229
y3 = 422252
'''
#cam3
x1 = -9669
y1 = -644
x2 = 1171
y2 = -615
x3 = 16
y3 = 4446

#print(x1, y1)
#print(x2, y2)
#print(x3, y3)

theta = (math.atan(y1/(x1+0.001)), math.atan(y2/(x2+0.001)), math.atan(y3/(x3+0.001)))
p = (math.sqrt(x1**2+y1**2), math.sqrt(x2**2+y2**2), math.sqrt(x3**2+y3**2))


'''
x1 sin(theta[0]) - y1 cos(theta[0]) = 0
x2
x3
-----------------
p1 = n1 f   //  p2 = n2 f   //  p3 = n3 f
'''
ncos1 = -(math.cos(theta[1] - theta[2])) / (math.cos(theta[0] - theta[1])) * math.cos(theta[2] - theta[0])
ncos2 = -(math.cos(theta[2] - theta[1])) / (math.cos(theta[0] - theta[1])) * math.cos(theta[1] - theta[2])
ncos3 = -(math.cos(theta[0] - theta[1])) / (math.cos(theta[1] - theta[2])) * math.cos(theta[2] - theta[0])
if ncos1 < 0 :
    ncos1 = -ncos1
if ncos2 < 0 :
    ncos2 = -ncos2
if ncos3 < 0 :
    ncos3 = -ncos3
n1 = math.sqrt(ncos1)
n2 = math.sqrt(ncos2)
n3 = math.sqrt(ncos3)

f1 = p[0]/n1
f2 = p[1]/n2
f3 = p[2]/n3
print(f1, f2, f3)
#f = (f1+f2+f3)/3
f = f3
print("f = ", f)

'''
x1 = p[0] cos(theta[0]) ,   y1 = p[0] sin(theta[0])
x2 = p[1] cos(theta[1]) ,   y2 = p[1] sin(theta[1])
x2 = p[2] cos(theta[2]) ,   y3 = p[2] sin(theta[2])
'''
#########################################
# su3 R 정보
'''
rvec_ = np.array(js["camera_rotation"])
rvec = np.array(rvec_).reshape(1, 3).astype(np.float32)
R_, _ = cv2.Rodrigues(rvec)
print("org R : ", R_)
'''

########################################
########################################
# roll, tilt로 구한 R !!!!!!!!!!!!!!!!!!!

'''
roll_down = np.sqrt((vpts[0][0]-vpts[1][0])**2 + (vpts[0][1]-vpts[1][1])**2)
roll_up = abs(vpts[0][1]-vpts[1][1])
roll = math.asin(roll_up/roll_down)
if (vpts[0][0]-vpts[1][0]) / (vpts[0][1]-vpts[1][1]) < 0 :
    roll = -roll

p1 = (vpts[0][0]/vpts[0][2], vpts[0][1]/vpts[0][2])
p2 = (vpts[1][0]/vpts[1][2], vpts[1][1]/vpts[1][2])
p3 = (vpts[2][0]/vpts[2][2], vpts[2][1]/vpts[2][2])
#tilt_d = np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1)
tilt_d = abs((p2[0]-p1[0])*(p1[1]-p3[1]) - (p1[0]-p3[0])*(p2[1]-p1[1])) / np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
tilt = math.atan(tilt_d/f)
'''

#*f*256+cx  *f*256+cy
p1 = (x1 , y1 )
p2 = (x2 , y2 )
p3 = (x3 , y3 )


roll_down = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
roll_up = abs(p1[1]-p2[1])
roll = math.asin(roll_up/roll_down)
if (p1[0]-p2[0]) / (p1[1]-p2[1]+0.001) < 0 :
    roll = -roll
print("roll : ", roll)
# 영상 원점 말고 vp Z축의 2D 점으로 구함
#tilt_d = np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1)
tilt_d = abs((p2[0]-p1[0])*(p1[1]-p3[1]) - (p1[0]-p3[0])*(p2[1]-p1[1])) / np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
tilt = math.atan(tilt_d/f)

R = [[math.cos(roll), -math.sin(roll)*math.cos(tilt), math.sin(roll)*math.sin(tilt)],
     [math.sin(roll), math.cos(roll)*math.cos(tilt), math.cos(roll)*-math.sin(tilt)],
     [0, math.sin(tilt), math.cos(tilt)]]
print("R-est : ", R[0], "\n", R[1], "\n", R[2])

K = [[f, 0, cx], [0, f, cy], [0,0,1]]
KR = np.dot(K, R)
print("KR_est : ", KR)
KR_inv = np.linalg.inv(KR)

########### uv상의 두길이 비교
# uv에 경첩 두개 넣어 길이 비교
# 위xy, 아래xy
#obj1 = (279, 18, 285, 52)      # 1300 이미지 두 경첩
#obj2 = (282, 283, 286, 308)
#obj1 = (12, 96, 28, 146)        # 870 이미지 왼위, 오른아래
#obj2 = (434, 309, 433, 339)
#obj1 = (209, 88, 212, 263)        # MOT02 왼쪽 창가들
#obj2 = (388, 210, 391, 329)
#obj1 = (220, 73, 253, 827)        # MOT02 가게 유리길이
#obj2 = (681, 158, 709, 761)


obj1 = (516, 395, 511, 546)        # MOT02 할머니 키
obj2 = (529, 374, 517, 527)         #cam3 남3키 200, 177, 184라고 추정
obj3 = (579, 405, 571, 540)
obj4 = (639, 403, 634, 544)


# obj는 물제 길이 (x1, y1, x2, y2)
def diff (obj1, obj2, KR_inv) :
    # 두 경첩의 4포인트를 XYZ로 변환
    Wobj1_t = np.dot(KR_inv, np.transpose([obj1[0], obj1[1], 1]))
    Wobj1_b = np.dot(KR_inv, np.transpose([obj1[2], obj1[3], 1]))
    Wobj2_t = np.dot(KR_inv, np.transpose([obj2[0], obj2[1], 1]))
    Wobj2_b = np.dot(KR_inv, np.transpose([obj2[2], obj2[3], 1]))
    #print(Wobj1_t, Wobj1_b, Wobj2_t, Wobj2_b)

    Wob1_t = [Wobj1_t[0]/Wobj1_t[2],Wobj1_t[1]/Wobj1_t[2]]
    Wob1_b = [Wobj1_b[0]/Wobj1_b[2],Wobj1_b[1]/Wobj1_b[2]]
    Wob2_t = [Wobj2_t[0]/Wobj2_t[2],Wobj2_t[1]/Wobj2_t[2]]
    Wob2_b = [Wobj2_b[0]/Wobj2_b[2],Wobj2_b[1]/Wobj2_b[2]]

    DIFF1 = np.sqrt((Wob1_t[0] - Wob1_b[0]) ** 2 + (Wob1_t[1] - Wob1_b[1]) ** 2)
    DIFF2 = np.sqrt((Wob2_t[0] - Wob2_b[0]) ** 2 + (Wob2_t[1] - Wob2_b[1]) ** 2)

    print("DIFF1 : ", DIFF1)  # X/Z, Y/Z 후 계산 >> 다른 칼리브도 비슷한 값을 가짐 1.0대
    print("DIFF2 : ", DIFF2)
    '''
    # XYZ 길이 파악
    Wobj1 = (Wobj1_t[0] - Wobj1_b[0], Wobj1_t[1] - Wobj1_b[1], Wobj1_t[2] - Wobj1_b[2])  # 위아래 포인트 X,Y,Z 차이값
    Wobj2 = (Wobj2_t[0] - Wobj2_b[0], Wobj2_t[1] - Wobj2_b[1], Wobj2_t[2] - Wobj2_b[2])
    #diff1 = np.sqrt(Wobj1[0] ** 2 + Wobj1[1] ** 2 + Wobj1[2] ** 2)  # 틀렸어
    #diff2 = np.sqrt(Wobj2[0] ** 2 + Wobj2[1] ** 2 + Wobj2[2] ** 2)
    DIFF1 = np.sqrt((Wobj1[0] / Wobj1[2]) ** 2 + (Wobj1[1] / Wobj1[2]) ** 2)  # 얘가 맞음
    DIFF2 = np.sqrt((Wobj2[0] / Wobj2[2]) ** 2 + (Wobj2[1] / Wobj2[2]) ** 2)


    #print("diff1 : ", diff1)  # XYZ 아님!!!!!!!!!!!!
    #print("diff2 : ", diff2)
    print("DIFF1 : ", DIFF1)  # 계산 후 X/Z, Y/Z
    print("DIFF2 : ", DIFF2)  # 1300경첩두길이 0.834923, 0.8326595
    # 670경첩두길이  1.195,  0.933
    '''
diff(obj1, obj2, KR_inv)
diff(obj3, obj4, KR_inv)




########################################################

## 선긋기 잘못됨 ##
'''
#with np.load("./data/su3/192/0053_0_label.npz") as npz:     # 원래 이미지랑 있던 vpt랑 비교
#    orgvpt = np.array([npz[d] for d in ["vpts"]])
#    print(orgvpt)
with np.load(rootdir + "frame-000670-vanish.npz") as npz:
    vpts = np.array([npz[d] for d in ["x", "y", "z"]])
    print(vpts)
    # vpts는 3 by 3
#print(vpts[0][0][0])

f = 2.1 #2.4
u = []
v = []
for w in vpts:   # SU3에선 vpts[0]을 range로 vpts["vpts_gt"]만 처리
    u.append(w[0] / w[2]*f*256+256)         # 점 이상함
    v.append(w[1] / w[2]*f*256+256)


uv = []
uv.append(u)
uv.append(v)
uv.append([1,1,1])
print(uv)


image = skimage.io.imread('./data/su3/192/0053_0.png')
image = skimage.transform.resize(image, (512, 512))
plt.imshow(image)
plt.scatter(uv[0], uv[1])
for i in range(len(uv[0])) :
    for xy in np.linspace(0, 512, 5):
        line = plt.plot(
            [uv[0][i], xy, uv[0][i], xy, uv[0][i], 0, uv[0][i], 511],
            [uv[1][i], 0, uv[1][i], 511, uv[1][i], xy, uv[1][i], xy],
        )
        plt.setp(line, linewidth=0.1)

plt.savefig('./result/eval/0053_0.png', dpi=300)
'''

'''
u0 = image.shape[0]/2
v0 = image.shape[1]/2

print(u[0]-u0)
print(u[1]-u0)
print(u[2]-u0)

print(v[0]-v0)
print(v[1]-v0)
print(v[2]-v0)
'''

