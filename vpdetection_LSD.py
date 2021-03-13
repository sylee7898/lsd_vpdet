from scipy import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
from collections import Counter

rootdir = '/home/seungyeon/Desktop/git/neurvps/data/line/'
#rootdir = 'E:/MMI/2020 ReID/multi-cam/Esing/Distance-based Topology/Calibration/CODE/neurvps/data/matlab_line/'

# MOT16-02-067
# MOT16-04-001
# MOT16-09-001
# MOT16-10-218
name = 'MOT16-09-001'
#name = 'camera1_bg'
''' 모든 이미지 파악 > 이미지 이름 따와 > 순서대로 실행 '''
img = cv2.imread(rootdir + name + '.jpg')


def get_crosspt(line1, line2):
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    if x12==x11 or x22==x21:
        return 0,0
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    #print("m1 : ", m1, "m2 : ", m2)
    if m1==m2:
        print('parallel')
        return 0,0
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11
    '''
    # 이렇게 구해도 같음 
    b1 = y12 - m1*x12
    b2 = y22 - m2*x22
    y = (b1*m2 - b2*m1) / (m2-m1)
    x = (y - b1) / m1
    '''

    # y축을 영상 중심으로부터 뒤집기
    #cy = img.shape[0] - cy
    #cx = img.shape[1] - cx

    return cx, cy


# 이미 구한 line 파일 받아서 vp정하기
def candidate_vps(img):

    mat_file = io.loadmat(rootdir + 'line/' + name + '_line.mat')

    lines = mat_file['lines']
    labels = mat_file['labels']

    ######################

    vp1x = []
    vp1y = []
    vp2x = []
    vp2y = []
    vp3x = []
    vp3y = []

    # labels = [[1][2][3][...]]
    # print(labels)

    label = [1, 2, 3]
    ##################################
    #print("라벨 빈도수 순서대로 : ", label)

    for LINE in range(0, len(lines) - 1):  # 라인 두개씩 비교하니까 두개씩 점프
        # for LABEL in range(1,2) :        # label 만 먼저 찾아보기
        if labels[LINE] == int(label[0]):
            line1 = lines[LINE]
            # cv2.line(img, (line1[0], line1[1]), (line1[2],line1[3]), 'red', 1)
            for LINE2 in range(LINE + 1, len(lines)):  # 그 이후라인들중 line2 있나 찾기
                if labels[LINE2] == int(label[0]):
                    line2 = lines[LINE + 1]
                    x, y = get_crosspt(line1, line2)
                    if (x<0 or x>img.shape[1]) and (y<0 or y>img.shape[0]):  # x,y가 none 아닐때, 영상내에 없을 때
                        if (abs(x) < 30000 and abs(y) < 30000) :
                            vp1x.append(x)
                            vp1y.append(y)
                    break
        if labels[LINE] == int(label[1]):
            line1 = lines[LINE]
            # cv2.line(img, (line1[0], line1[1]), (line1[2],line1[3]), 'red', 1)
            for LINE2 in range(LINE + 1, len(lines)):  # 그 이후라인들중 line2 있나 찾기
                if labels[LINE2] == int(label[1]):
                    line2 = lines[LINE + 1]
                    x, y = get_crosspt(line1, line2)
                    if (x<0 or x>img.shape[1]) and (y<0 or y>img.shape[0]):  # x,y가 none 아닐때, 영상내에 없을 때
                        if (abs(x) < 50000 and abs(y) < 50000):
                            vp2x.append(x)
                            vp2y.append(y)
                    break
        if labels[LINE] == int(label[2]):
            line1 = lines[LINE]
            # cv2.line(img, (line1[0], line1[1]), (line1[2],line1[3]), 'red', 1)
            for LINE2 in range(LINE + 1, len(lines)):  # 그 이후라인들중 line2 있나 찾기
                if labels[LINE2] == int(label[2]):
                    line2 = lines[LINE + 1]
                    x, y = get_crosspt(line1, line2)
                    #if (x<0 or x>img.shape[1]) and (y<0 or y>img.shape[0]):  # x,y가 none 아닐때, 영상내에 없을 때
                    if (x < 0 or x > img.shape[1]) and (y < 0 or y > img.shape[0]):  # x,y가 none 아닐때, 영상내에 없을 때
                        if (abs(x) < 30000 and abs(y) < 30000):
                            vp3x.append(x)
                            vp3y.append(y)
                    break

    sum1x = 0
    sum1y = 0
    sum2x = 0
    sum2y = 0
    sum3x = 0
    sum3y = 0
    for i in range(len(vp1x)):
        sum1x += vp1x[i]
        sum1y += vp1y[i]

    for i in range(len(vp2x)):
        sum2x += vp2x[i]
        sum2y += vp2y[i]
    for i in range(len(vp3x)):
        sum3x += vp3x[i]
        sum3y += vp3y[i]


    VP1 = (sum1x / len(vp1x), sum1y / len(vp1x))
    VP2 = (sum2x / len(vp2x), sum2y / len(vp2x))
    VP3 = (sum3x / len(vp3x), sum3y / len(vp3x))

    plt.imshow(img)
    for i in range(len(vp1x)):
        plt.scatter(vp1x[i], vp1y[i], s=0.2, color='blue')
    for i in range(len(vp2x)):
        plt.scatter(vp2x[i], vp2y[i], s=0.2, color='green')
    for i in range(len(vp3x)):
        plt.scatter(vp3x[i], vp3y[i], s=0.2, color='red')

    #plt.scatter(VP1[0], VP1[1], s=2, color='black')
    #plt.scatter(VP2[0], VP2[1], s=2, color='black')
    #plt.scatter(VP3[0], VP3[1], s=2, color='black')
    '''
    plt.scatter(-4863, 630, s=3, color='yellow')
    plt.scatter(727, 467, s=3, color='yellow')
    plt.scatter(1206, 23776, s=3, color='yellow')
    '''
    plt.scatter(204, -16510, s=3, color='yellow')
    plt.scatter(-516.88, 656.77, s=3, color='yellow')
    plt.scatter(2565.69, 495.98, s=3, color='yellow')

    print(VP1)
    print(VP2)
    print(VP3)

    plt.show()
    # plt.savefig('./result/vpdet/'+name+'.png', img)





candidate_vps(img)






################################################################################
####################                               #############################
####################         L S D                 #############################
####################                               #############################
################################################################################

M_PI = 3.14159265358979323846

def error(msg) :
    # msg : char
    print(stderr,"LSD Error: %s\n",msg)
    exit(EXIT_FAILURE)


def angle_diff(a, b) :
    a -= b
    while( a <= -M_PI ) : a += 2*M_PI
    while( a >   M_PI ) : a -= 2*M_PI
    if( a < 0.0 ) : a = -a
    return a



'''( struct point * reg, int reg_size, double x, double y,
     image_double modgrad, double reg_angle, double prec )
'''
def get_theta (reg, reg_size, x, y, modgrad, reg_angle, prec) :

    Ixx = 0.0
    Iyy = 0.0
    Ixy = 0.0


    # check parameters
    if( reg == NULL ) : print("get_theta: invalid region.")
    if( reg_size <= 1 ) : print("get_theta: region size <= 1.")
    if( modgrad == NULL or modgrad.data == NULL ) : print("get_theta: invalid 'modgrad'.")
    if( prec < 0.0 ) : print("get_theta: 'prec' must be positive.")

    #compute inertia matrix
    for i in range(0, reg_size) :
        weight = modgrad.data[ reg[i].x + reg[i].y * modgrad.xsize ]
        Ixx += ( (double) (reg[i].y - y) ) * ( (double) (reg[i].y - y) ) * weight
        Iyy += ( (double) (reg[i].x - x) ) * ( (double) (reg[i].x - x) ) * weight
        Ixy -= ( (double) (reg[i].x - x) ) * ( (double) (reg[i].y - y) ) * weight

    if( double_equal(Ixx,0.0) and double_equal(Iyy,0.0) and double_equal(Ixy,0.0) ) :
        print("get_theta: null inertia matrix.")
        return 0

    # compute smallest eigenvalue
    lambda_ = 0.5 * ( Ixx + Iyy - sqrt( (Ixx-Iyy)*(Ixx-Iyy) + 4.0*Ixy*Ixy ) )

    # compute angle
    if (fabs(Ixx) > fabs(Iyy)) :
        theta = math.atan2(lambda_-Ixx, Ixy)
    else :  math.atan2(Ixy, lambda_-Iyy)

    '''
    The previous procedure don't cares about orientation,
    so it could be wrong by 180 degrees. Here is corrected if necessary.
    '''

    if( angle_diff(theta, reg_angle) > prec ) : theta += M_PI

    return theta




'''
( struct point * reg, int reg_size,
  image_double modgrad, double reg_angle,
  double prec, double p, struct rect * rec )  :
  
  double x,y,dx,dy,l,w,theta,weight,sum,l_min,l_max,w_min,w_max;
  int i;
'''
def refion2rect(reg, reg_size, modgrad, reg_angle, prec, p ,rec) :
    '''
    :param reg: struct point *
    :param reg_size: int
    :param modgrad: image_double
    :param reg_angle: double
    :param prec: double
    :param p: double
    :param rec: struct rect *
    :return:
    '''

    ''' check parameters '''
    if ( reg == NULL ) :
        print("region2rect: invalid region.")
        return 0
    if ( reg_size <= 1 ) :
        print("region2rect: region size <= 1.")
        return 0
    if ( modgrad == NULL or modgrad.data == NULL ) :
        print("region2rect: invalid image 'modgrad'.")
        return 0
    if ( rec == NULL ) :
        print("region2rect: invalid 'rec'.")
        return 0

    '''
    center of the region:

     It is computed as the weighted sum of the coordinates
     of all the pixels in the region. The norm of the gradient
     is used as the weight of a pixel. The sum is as follows:
       cx = \sum_i G(i).x_i
       cy = \sum_i G(i).y_i
     where G(i) is the norm of the gradient of pixel i
     and x_i,y_i are its coordinates.
    '''
    x = y = sum = 0.0
    for i in range(0, reg,size) :
        weight = modgrad.data[ reg[i].x + reg[i].y * modgrad.xsize ]
        x += (double) (reg[i].x * weight)
        y += (double) (reg[i].y * weight)
        sum += weight

    if( sum <= 0.0 ) :
        print("region2rect: weights sum equal to zero.")
        return 0
    x /= sum
    y /= sum

    # theta
    theta = get_theta(reg,reg_size,x,y,modgrad,reg_angle,prec);

    '''
    length and width:

     'l' and 'w' are computed as the distance from the center of the
     region to pixel i, projected along the rectangle axis (dx,dy) and
     to the orthogonal axis (-dy,dx), respectively.

     The length of the rectangle goes from l_min to l_max, where l_min
     and l_max are the minimum and maximum values of l in the region.
     Analogously, the width is selected from w_min to w_max, where
     w_min and w_max are the minimum and maximum of w for the pixels
     in the region.
    '''
    dx = cos(theta)
    dy = sin(theta)
    l_min = l_max = w_min = w_max = 0.0
    for i in range(0,reg_size) :

        l =  ( (double)(reg[i].x - x)) * dx + ( (double)(reg[i].y - y)) * dy
        w = -( (double)(reg[i].x - x)) * dy + ( (double)(reg[i].y - y)) * dx

        if( l > l_max ) : l_max = l
        if( l < l_min ) : l_min = l
        if( w > w_max ) : w_max = w
        if( w < w_min ) : w_min = w

    #store values
    rec.x1 = x + l_min * dx
    rec.y1 = y + l_min * dy
    rec.x2 = x + l_max * dx
    rec.y2 = y + l_max * dy
    rec.width = w_max - w_min
    rec.x = x
    rec.y = y
    rec.theta = theta
    rec.dx = dx
    rec.dy = dy
    rec.prec = prec
    rec.p = p

    '''
    we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
    '''
    if( rec.width < 1.0 ) : rec.width = 1.0


#############################################################
'''
static image_double ll_angle( image_double in, double threshold,
                              struct coorlist ** list_p, void ** mem_p,
                              image_double * modgrad, unsigned int n_bins,
                              double max_grad )
'''
def ll_angle (input, threshold, list_p, mem_p, modgrad, n_bins, max_grad) :
    # gradient 관련
    '''
    image_double g;
    unsigned int n,p,x,y,adr,i;
    double com1,com2,gx,gy,norm,norm2;
    '''
    '''
    the rest of the variables are used for pseudo-ordering
    the gradient magnitude values
    '''

    list_count = 0
    '''
    struct coorlist * list
    struct coorlist ** range_l_s    # array of pointers to start of bin list
    struct coorlist ** range_l_e    # array of pointers to end of bin list
    struct coorlist * start
    struct coorlist * end
    '''

    # check parameters
    if( input == NULL or input.data == NULL or input.xsize == 0 or input.ysize == 0 ) :
        error("ll_angle: invalid image.")
    if( threshold < 0.0 ) : error("ll_angle: 'threshold' must be positive.")
    if( list_p == NULL ) : error("ll_angle: NULL pointer 'list_p'.")
    if( mem_p == NULL ) : error("ll_angle: NULL pointer 'mem_p'.")
    if( modgrad == NULL ) : error("ll_angle: NULL pointer 'modgrad'.")
    if( n_bins == 0 ) : error("ll_angle: 'n_bins' must be positive.")
    if( max_grad <= 0.0 ) : error("ll_angle: 'max_grad' must be positive.")

    #image size shortcuts
    n = input.ysize
    p = input.xsize

    # allocate output image
    g = new_image_double(input.xsize,input.ysize)

    # get memory for the image of gradient modulus
    modgrad = new_image_double(input.xsize,input.ysize)

    # get memory for "ordered" list of pixels
    '''
    list = (struct coorlist *) calloc( (size_t) (n*p), sizeof(struct coorlist) )
    mem_p = (void *) list
    range_l_s = (struct coorlist **) calloc( (size_t) n_bins, sizeof(struct coorlist *) )
    range_l_e = (struct coorlist **) calloc( (size_t) n_bins, sizeof(struct coorlist *) )
    '''
    list_= []
    mem_p = []
    range_l_s = []
    range_l_e = []
    if( list_ == NULL or range_l_s == NULL or range_l_e == NULL ) :
        error("not enough memory.")
    for i in range(0, n_bins) :
        range_l_s[i] = range_l_e[i] = NULL

    # 'undefined' on the down and right boundaries
    for x in range(0,p) :   g.data[(n-1)*p+x] = NOTDEF
    for y in range(0,n) :   g.data[p*y+p-1]   = NOTDEF

    # compute gradient on the remaining pixels
    for x in range(0, p-1) :
        for y in range(0, n-1) :
            adr = y*p+x

            '''
            Norm 2 computation using 2x2 pixel window:
               A B
               C D
            and
               com1 = D-A,  com2 = B-C.
            Then
               gx = B+D - (A+C)   horizontal difference
               gy = C+D - (A+B)   vertical difference
            com1 and com2 are just to avoid 2 additions.
            '''

            com1 = input.data[adr+p+1] - input.data[adr]
            com2 = input.data[adr+1]   - input.data[adr+p]

            gx = com1+com2  # gradient x component
            gy = com1-com2  # gradient y component
            norm2 = gx*gx+gy*gy
            norm = np.sqrt( norm2 / 4.0 )  # gradient norm

            modgrad.data[adr] = norm    # store gradient norm

            if( norm <= threshold ) :   # norm too small, gradient no defined
                g.data[adr] = NOTDEF   # gradient angle not defined
            else :
                # gradient angle computation
                g.data[adr] = atan2(gx,-gy)

                # store the point in the right bin according to its norm
                i = (unit) (norm) * (double) (n_bins / max_grad)
                if( i >= n_bins ) : i = n_bins-1
                if( range_l_e[i] == NULL ) :
                    range_l_s[i] = range_l_e[i] = (list_+list_count) +1
                else :
                    range_l_e[i].next = list_+list_count
                    range_l_e[i] = (list_+list_count) +1

                range_l_e[i].x = (int) (x)
                range_l_e[i].y = (int) (y)
                range_l_e[i].next = NULL


    '''
    Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with higher gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
    '''
    for i in range(n_bins-1, 0, -1) :   # (i=n_bins-1; i>0 && range_l_s[i]==NULL; i--);
        if(range_l_s[i] == NULL) :
            start = range_l_s[i]
            end = range_l_e[i]
        if( start != NULL ) :
            for i in range(i,0,-1) :      # (i--;i>0; i--)
                if( range_l_s[i] != NULL ) :
                    end.next = range_l_s[i]
                    end = range_l_e[i]
    list_p = start

    # free memory
    #free( (void *) range_l_s )
    #free( (void *) range_l_e )

    return g



#############################################################
'''
ntuple_list LineSegmentDetection( image_double image, double scale,
                                  double sigma_scale, double quant,
                                  double ang_th, double eps, double density_th,
                                  int n_bins, double max_grad,
                                  image_int * region )
'''
'''
def LineSegmentDetection ( image, scale, sigma_scale, quant, ang_th, eps, density_th, \
                            n_bins, max_grad,region ) :

    ntuple_list out = new_ntuple_list(5);     // dim=5, [0,1] / malloc = maxsize(=1)*size
    image_double scaled_image,angles,modgrad;
    image_char used;
    struct coorlist * list_p;
    void * mem_p;
    struct rect rec;
    struct point * reg;
    int reg_size,min_reg_size,i;
    unsigned int xsize,ysize;
    double rho,reg_angle,prec,p,log_nfa,logNT;
    int ls_count = 0;                   /* line segments are numbered 1,2,3,... */


    # check parameters
    if( image==NULL || image->data==NULL || image->xsize==0 || image->ysize==0 )
        error("invalid image input.");
    if( scale <= 0.0 ) error("'scale' value must be positive.");
    if( sigma_scale <= 0.0 ) error("'sigma_scale' value must be positive.");
    if( quant < 0.0 ) error("'quant' value must be positive.");
    if( ang_th <= 0.0 || ang_th >= 180.0 )
        error("'ang_th' value must be in the range (0,180).");
    if( density_th < 0.0 || density_th > 1.0 )
        error("'density_th' value must be in the range [0,1].");
    if( n_bins <= 0 ) error("'n_bins' value must be positive.");
    if( max_grad <= 0.0 ) error("'max_grad' value must be positive.");


    /* angle tolerance */
    prec = M_PI * ang_th / 180.0;
    p = ang_th / 180.0;
    rho = quant / sin(prec); /* gradient magnitude threshold */


    /* scale image (if necessary) and compute angle at each pixel */
    if( scale != 1.0 )
        {
          scaled_image = gaussian_sampler( image, scale, sigma_scale );
          angles = ll_angle( scaled_image, rho, &list_p, &mem_p,
                             &modgrad, (unsigned int) n_bins, max_grad );
          free_image_double(scaled_image);
        }
    else
        angles = ll_angle( image, rho, &list_p, &mem_p, &modgrad, (unsigned int) n_bins, max_grad );
    xsize = angles->xsize;
    ysize = angles->ysize;
    logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0;
    min_reg_size = (int) (-logNT/log10(p)); /* minimal number of points in region
                                                 that can give a meaningful event */


    /* initialize some structures */
    if( region != NULL ) /* image to output pixel region number, if asked */
        *region = new_image_int_ini(angles->xsize,angles->ysize,0);
    used = new_image_char_ini(xsize,ysize,NOTUSED);
    reg = (struct point *) calloc( (size_t) (xsize*ysize), sizeof(struct point) );
    if( reg == NULL ) error("not enough memory!");


    /* search for line segments */
    for(; list_p != NULL; list_p = list_p->next )
        if( used->data[ list_p->x + list_p->y * used->xsize ] == NOTUSED &&
            angles->data[ list_p->x + list_p->y * angles->xsize ] != NOTDEF )
           /* there is no risk of double comparison problems here
              because we are only interested in the exact NOTDEF value */
          {
            /* find the region of connected point and ~equal angle */
            region_grow( list_p->x, list_p->y, angles, reg, &reg_size,
                         &reg_angle, used, prec );

            /* reject small regions */
            if( reg_size < min_reg_size ) continue;

            /* construct rectangular approximation for the region */
            region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&rec);

            /* Check if the rectangle exceeds the minimal density of
            region points. If not, try to improve the region.
            The rectangle will be rejected if the final one does
            not fulfill the minimal density condition.
            This is an addition to the original LSD algorithm published in
            "LSD: A Fast Line Segment Detector with a False Detection Control"
            by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
            The original algorithm is obtained with density_th = 0.0.
            */
            if( !refine( reg, &reg_size, modgrad, reg_angle,
                         prec, p, &rec, used, angles, density_th ) ) continue;

            /* compute NFA value */
            log_nfa = rect_improve(&rec,angles,logNT,eps);
            if( log_nfa <= eps ) continue;

            /* A New Line Segment was found! */
            ++ls_count;  /* increase line segment counter */

            /*
               The gradient was computed with a 2x2 mask, its value corresponds to
               points with an offset of (0.5,0.5), that should be added to output.
               The coordinates origin is at the center of pixel (0,0).
             */
            rec.x1 += 0.5; rec.y1 += 0.5;
            rec.x2 += 0.5; rec.y2 += 0.5;

            /* scale the result values if a subsampling was performed */
            if( scale != 1.0 )
              {
                rec.x1 /= scale; rec.y1 /= scale;
                rec.x2 /= scale; rec.y2 /= scale;
                rec.width /= scale;
              }

            /* add line segment found to output */
            add_5tuple(out, rec.x1, rec.y1, rec.x2, rec.y2, rec.width);

            /* add region number to 'region' image if needed */
            if( region != NULL )
              for(i=0; i<reg_size; i++)
                (*region)->data[reg[i].x+reg[i].y*(*region)->xsize] = ls_count;
          }


    /* free memory */
    free_image_double(angles);
    free_image_double(modgrad);
    free_image_char(used);
    free( (void *) reg );
    free( (void *) mem_p );

    return out

'''
##########################################################







