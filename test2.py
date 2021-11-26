# %%
import numpy as np
import matplotlib.pyplot as plt
# xx=np.array([0,150,250,350, 200,110, 20,0])
# yy=np.array([340,440,360,-20,-70 ,-120, 80,340])

xx=np.array([0,150,250,320, 200,110, 20,0])
yy=np.array([340,440,360,-60,-90 ,-120, 80,340])




# yy = np.array([300, 300, 330, 10, 20, 300])
# plt.figure(dpi=300,figsize=(3,3))
# plt.plot(xx,yy)

# 计算斜率函数
def slope_cal(x_cor,y_cor):
    xx=x_cor

    yy=y_cor
    xx_dif=np.diff(xx)
    yy_dif=np.diff(yy)
    slope=yy_dif/xx_dif
    slope=np.append(slope,slope[0])
    return slope

slope=slope_cal(xx,yy)    

#  逆时针旋转90°
# max_x=np.max(xx)
# max_y=np.max(yy)
# min_x=np.min(xx)
# min_y=np.min(yy)
# mid_x=(max_x+min_x)/2
# mid_y=(max_y+min_y)/2


# xx1=[]
# yy1=[]
# for ii in range(len(xx )):
#     r=np.sqrt((xx[ii]-mid_x)**2+(yy[ii]-mid_y)**2)
#     x=(xx[ii]-mid_x)*np.cos(np.pi/2)-(yy[ii]-mid_y)*np.sin(np.pi/2)+mid_x
#     y=(xx[ii]-mid_x)*np.sin(np.pi/2)-(yy[ii]-mid_y)*np.cos(np.pi/2)+mid_y
#     xx1.append(x)
#     yy1.append(y)

# xx1=np.array(xx1)
# yy1=np.array(yy1)
# plt.plot(xx1,yy1)
# plt.show()



def edgeAndCut(xx, yy):
    """
        Return the edge set and cut set of polygon
        :param xx: x axis set
        :param yy: y axis set
        :return: edge: edge set
                 cut: cut set
        """
    # Initialization
    edge = np.zeros((xx.shape[0] - 1))
    cut = np.zeros((xx.shape[0] - 1))
    xx_dif = np.diff(xx)
    yy_dif = np.diff(yy)
    # Calculate the edge
    for i in range(edge.shape[0]):
        # edge[i] is the length distance between (xx[i], yy[i]) and (xx[i+1], yy[i+1])
        edge[i] = np.sqrt(xx_dif[i] ** 2 + yy_dif[i] ** 2)

    # Calculate the cut
    for i in range(cut.shape[0]):
        # cut[i] is the length of the cut between (xx[i+2], yy[i+2]) and (xx[i], yy[i])
        if i == cut.shape[0] - 1:
            cut[i] = np.sqrt((xx[i] - xx[1]) ** 2 + (yy[i] - yy[1]) ** 2)
        else:
            cut[i] = np.sqrt((xx[i] - xx[i + 2]) ** 2 + (yy[i] - yy[i + 2]) ** 2)

    return edge, cut

def delVertex(xx, yy, threshold=0.005):
    """
        Delete points that do not meet the requirements
        :param xx: x axis set
        :param yy: y axis set
        :param threshold: threshold condition
        :return: xx-x axis set of polygon after delete
                    yy-y axis set of polygon after delete
        """

    edge, cut = edgeAndCut(xx, yy)
    ratio = []  # ratio of 2 adjacent edges and the corresponding cut
    mid_point = []
    for i in range(cut.shape[0]):
        if i == cut.shape[0] - 1:
            ratio.append(abs(edge[i] + edge[0] - cut[i]) / (edge[i] + edge[0]))
            mid_point.append([1 / 2 * (xx[1] - xx[i]), 1 / 2 * (yy[1] - yy[i])])
        else:
            ratio.append(abs(edge[i] + edge[i + 1] - cut[i]) / (edge[i] + edge[i + 1]))
            mid_point.append([1 / 2 * (xx[i + 2] - xx[i]), 1 / 2 * (yy[i + 2] - yy[i])])

    del_pos = []  # index of cord to delete
    for i in range(len(ratio)):
        # if isInPloygon(xx, yy, mid_point[i][0], mid_point[i][1]):
            if ratio[i] < threshold or ratio[i] > 1 - threshold:
                # del the vertex whose ratio is less than threshold
                del_pos.append(i + 1)

    # process of deleting cord
    temp = 0
    for x in del_pos:
        xx = np.delete(xx, x - temp)
        yy = np.delete(yy, x - temp)
        temp += 1

    # Reform xx, yy if necessary
    if xx[-1] != xx[0]:
        xx = np.insert(xx, 0, xx[-1])
        yy = np.insert(yy, 0, yy[-1])

    return xx, yy

def scale(x, y, sec_dis):
    """
        :param x: x axis set
        :param y: y axis set
        :param sec_dis: scaled distance
        :return: Scaled polygon point set
        """
    data = []
    for i in range(len(x) - 1):
        data.append([x[i], y[i]])
    num = len(data)
    xx = []
    yy = []
    for i in range(num):
        x1 = data[(i) % num][0] - data[(i - 1) % num][0]
        y1 = data[(i) % num][1] - data[(i - 1) % num][1]
        x2 = data[(i + 1) % num][0] - data[(i) % num][0]
        y2 = data[(i + 1) % num][1] - data[(i) % num][1]

        d_A = (x1 ** 2 + y1 ** 2) ** 0.5
        d_B = (x2 ** 2 + y2 ** 2) ** 0.5

        Vec_Cross = (x1 * y2) - (x2 * y1)
        if (d_A * d_B == 0):
            continue
        sin_theta = Vec_Cross / (d_A * d_B)
        if (sin_theta == 0):
            continue
        dv = sec_dis / sin_theta

        v1_x = (dv / d_A) * x1
        v1_y = (dv / d_A) * y1

        v2_x = (dv / d_B) * x2
        v2_y = (dv / d_B) * y2

        PQ_x = v1_x - v2_x
        PQ_y = v1_y - v2_y

        Q_x = data[(i) % num][0] + PQ_x
        Q_y = data[(i) % num][1] + PQ_y
        xx.append(Q_x)
        yy.append(Q_y)
    xx.append(xx[0])
    yy.append(yy[0])

    return xx, yy

plt.plot(xx, yy)
xx, yy = delVertex(xx, yy, threshold = 0.005)
plt.plot(xx, yy)
scale_x, scale_y = scale(xx, yy, sec_dis = 10)
plt.plot(scale_x, scale_y)
# plt.show()




def calculation_SN(xx, yy, ymax):
    # xx scaled x
    # yy scaled y
    y_max = ymax
    # y_min = np.min(yy)
    dya = np.array(yy) - y_max

    intersec_x = []
    intersec_y = []
    for ii in range(len(dya) - 1):
        mult = dya[ii]*dya[ii + 1]
        if mult < 0:
            intersec_x.append(xx[ii])
            intersec_x.append(xx[ii + 1])
            intersec_y.append(yy[ii])
            intersec_y.append(yy[ii + 1])

    slope_intersec = []
    const_intersec = []

    for ii in list(range(len(intersec_x))):
        if ii % 2 != 0:
            temp = (intersec_y[ii]-intersec_y[ii - 1])/(intersec_x[ii] - intersec_x[ii - 1])
            slope_intersec.append(temp)
            temp1 = intersec_y[ii] - temp*intersec_x[ii]
            const_intersec.append(temp1)

    if intersec_x[1] < intersec_x[2]:     #First point is to the left of the highest point

#
        if slope_intersec[0] > 0 and slope_intersec[1] < 0:    # Left slope is greater than 0, right slope is less than 0
            intersec_posi_x1 = (y_max - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2  = (y_max - const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] > 0 and slope_intersec[1] > 0:    # Left and right slope are greater than 0
            intersec_posi_x1 = (y_max - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2 = (y_max - 12-const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] < 0:    # Left and right slope are less than 0
            intersec_posi_x1 = (y_max - 12 - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2 = (y_max - const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] > 0:    # Left slope is less than 0, right slope is greater than 0
            intersec_posi_x1 = (y_max - 12 - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2 = (y_max - 12 - const_intersec[1])/slope_intersec[1]
        return intersec_posi_x1, intersec_posi_x2

    elif intersec_x[1] > intersec_x[2]:     #First point is to the right of the highest point

#
        if slope_intersec[0] > 0 and slope_intersec[1] < 0:    # Left slope is greater than 0, right slope is less than 0
            intersec_posi_x1 = (y_max - 12 - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2 = (y_max - 12 - const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] > 0 and slope_intersec[1] > 0:    # Left and right slope are greater than 0
            intersec_posi_x1 = (y_max - 12 - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2 = (y_max - const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] < 0:    # Left slope is less than 0
            intersec_posi_x1 = (y_max - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2 = (y_max - 12 - const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] > 0:    # Left slope is less than 0, right slope is greater than 0
            intersec_posi_x1 = (y_max - const_intersec[0])/slope_intersec[0]
            intersec_posi_x2 = (y_max - const_intersec[1])/slope_intersec[1]

        return intersec_posi_x2, intersec_posi_x1

    elif intersec_x[1] == intersec_x[2]:  #Intersect two adjacent lines

        if intersec_x[0] < intersec_x[3]:

            if slope_intersec[0] > 0 and slope_intersec[1] < 0:
                intersec_posi_x1 = (y_max - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - const_intersec[1]) / slope_intersec[1]

            elif slope_intersec[0] > 0 and slope_intersec[1] > 0:
                intersec_posi_x1 = (y_max - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - 12 - const_intersec[1]) / slope_intersec[1]

            elif slope_intersec[0] < 0 and slope_intersec[1] < 0:
                intersec_posi_x1 = (y_max - 12 - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - const_intersec[1]) / slope_intersec[1]

            elif slope_intersec[0] < 0 and slope_intersec[1] > 0:
                intersec_posi_x1 = (y_max - 12 - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - 12 - const_intersec[1]) / slope_intersec[1]

            return intersec_posi_x1, intersec_posi_x2

        elif intersec_x[0] > intersec_x[3]:

            if slope_intersec[0] > 0 and slope_intersec[1] < 0:
                intersec_posi_x1 = (y_max - 12 - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - 12 - const_intersec[1]) / slope_intersec[1]

            elif slope_intersec[0] > 0 and slope_intersec[1] > 0:
                intersec_posi_x1 = (y_max - 12 - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - const_intersec[1]) / slope_intersec[1]

            elif slope_intersec[0] < 0 and slope_intersec[1] < 0:
                intersec_posi_x1 = (y_max - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - 12 - const_intersec[1]) / slope_intersec[1]

            elif slope_intersec[0] < 0 and slope_intersec[1] > 0:
                intersec_posi_x1 = (y_max - const_intersec[0]) / slope_intersec[0]
                intersec_posi_x2 = (y_max - const_intersec[1]) / slope_intersec[1]

            return intersec_posi_x2, intersec_posi_x1






# Calculation length combination
len_build = np.array([7, 9, 11, 15])
len_2build = []
for ii in len_build:
    for jj in len_build:
        len_2build.append(ii + jj)
len_2build = np.unique(len_2build)

len_3build = []
for ii in len_build:
    for jj in len_build:
        for kk in len_build:
            len_3build.append(ii + jj + kk)
len_3build = np.unique(len_3build)

len_4build = []
for ii in len_build:
    for jj in len_build:
        for kk in len_build:
            for ll in len_build:
                len_4build.append(ii + jj + kk + ll)
len_4build = np.unique(len_4build)

len_build_total = np.hstack((len_2build, len_3build, len_4build))
len_build_total = np.sort(len_build_total[(len_build_total > 18) & (len_build_total < 50)])
len_build_total = np.unique(len_build_total)




def calculation_SN1(xx, yy, x_point):
    # xx scaled x
    # yy scaled y
    # y_max = ymax
    # y_min = np.min(yy)
    dya = np.array(xx) - x_point

    intersec_x = []
    intersec_y = []
    for ii in range(len(dya) - 1):
        mult = dya[ii]*dya[ii + 1]
        if mult < 0:
            intersec_x.append(xx[ii])
            intersec_x.append(xx[ii + 1])
            intersec_y.append(yy[ii])
            intersec_y.append(yy[ii + 1])

    slope_intersec = []
    const_intersec = []

    for ii in range(len(intersec_x)):
        if ii % 2 != 0:
            temp = (intersec_y[ii]-intersec_y[ii - 1])/(intersec_x[ii] - intersec_x[ii - 1])
            slope_intersec.append(temp)
            temp1 = intersec_y[ii] - temp*intersec_x[ii]
            const_intersec.append(temp1)

    if intersec_y[1] > intersec_y[2]:     #前两个点在上方

#
        if slope_intersec[0] > 0 and slope_intersec[1] < 0:    # Left slope is greater than 0, right slope is less than 0
            intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]

        elif slope_intersec[0] > 0 and slope_intersec[1] > 0:    # Left and right slope are greater than 0
            intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = (x_point ) * slope_intersec[1] + const_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] < 0:    # Left and right slope are less than 0
            intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] > 0:    # Left slope is less than 0, right slope is greater than 0
            intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = (x_point ) * slope_intersec[1] + const_intersec[1]
        return intersec_posi_y1, intersec_posi_y2

    elif intersec_y[1]< intersec_y[2]:     #前两个点在下方

#
        if slope_intersec[0] > 0 and slope_intersec[1] < 0:    # Left slope is greater than 0, right slope is less than 0
            intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = (x_point) * slope_intersec[1] + const_intersec[1]

        elif slope_intersec[0] > 0 and slope_intersec[1] > 0:    # Left and right slope are greater than 0
            intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] < 0:    # Left slope is less than 0
            intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = (x_point ) * slope_intersec[1] + const_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] > 0:    # Left slope is less than 0, right slope is greater than 0
            intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
            intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]

        return intersec_posi_y2, intersec_posi_y1

    elif intersec_y[1] == intersec_y[2]:  #Intersect two adjacent lines

        if intersec_y[0] > intersec_y[3]:

            if slope_intersec[0] > 0 and slope_intersec[1] < 0:
                intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = (x_point ) * slope_intersec[1] + const_intersec[1]

            elif slope_intersec[0] > 0 and slope_intersec[1] > 0:
                intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]

            elif slope_intersec[0] < 0 and slope_intersec[1] < 0:
                intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = (x_point ) * slope_intersec[1] + const_intersec[1]

            elif slope_intersec[0] < 0 and slope_intersec[1] > 0:
                intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]


            return intersec_posi_y1, intersec_posi_y2

        elif intersec_y[0] < intersec_y[3]:

            if slope_intersec[0] > 0 and slope_intersec[1] < 0:
                intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]

            elif slope_intersec[0] > 0 and slope_intersec[1] > 0:
                intersec_posi_y1 = x_point * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = (x_point ) * slope_intersec[1] + const_intersec[1]

            elif slope_intersec[0] < 0 and slope_intersec[1] < 0:
                intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = x_point * slope_intersec[1] + const_intersec[1]
            elif slope_intersec[0] < 0 and slope_intersec[1] > 0:
                intersec_posi_y1 = (x_point ) * slope_intersec[0] + const_intersec[0]
                intersec_posi_y2 = (x_point ) * slope_intersec[1] + const_intersec[1]

            return intersec_posi_y2, intersec_posi_y1



def closest(mylist, Number):
    # Results of two straight lengths
  """
    :param mylist: Unfilled building length
    :param Number: Remainder of the ground after the row
    :return: Row length and remaining land
    """
  answer = Number - 5 - mylist
  if all(answer < 0) == True:
        return [0, Number]
  else:
        answer1 = answer[(answer >= 0)]
        answer1 = min(answer1)
        indd = np.where(answer == answer1)
        return [mylist[indd[0]], answer1]

# ll=325  # as example
def arr_ew(ll, ll_build):
    # ll is the East-West length, ll_ Build is the length combination of all buildings
    sche = {}

    for ii in np.flipud(ll_build):
        times = ii - ll_build[0]
        if ll > (ii + 2.5)*2:
            ll1 = ll - (ii + 2.5)*2
            build_num = np.floor(ll1/(ii + 5))
            mod_num = np.mod(ll1, (ii + 5))
            if mod_num < min(ll_build):
                sche_name = str(ii)
                sche[sche_name] = [ii, build_num + 2, mod_num, 2*2.5 + build_num*5 + mod_num]
            else:
                aa = closest(ll_build[:times - 1], mod_num)
                sche_name = str(ii)
                sche[sche_name] = [ii, build_num + 2, mod_num, aa[0], aa[1], 2*2.5 + (build_num+1)*5 + aa[1]]

        if ll < (ii + 2.5)*2  and  ll > ii + 5:
            ll1 = ll - ii
            build_num = np.floor(ll1/(ii + 5))
            mod_num = np.mod(ll1, (ii + 5))
            if mod_num < min(ll_build):
                sche_name = str(ii)
                sche[sche_name] = [ii, build_num + 1, mod_num, 2*2.5 + build_num*5 + mod_num]
            else:
                aa = closest(ll_build[:times - 1], mod_num)
                sche_name = str(ii)
                sche[sche_name] = [ii,build_num + 1, mod_num, aa[0], aa[1], 2*2.5 + (build_num + 1)*5 + aa[1]]

    return sche



# %%
y_up=[]
y_down=[]
x_lengh=[]
plt.plot(scale_x, scale_y)
x_min=min(scale_x)
for ii in np.linspace(0.01,20,20000):
    x_min=x_min+ii
    intersec_posi_y1, intersec_posi_y2 = calculation_SN1(scale_x, scale_y, x_min)
    ll1 = np.abs(intersec_posi_y1 - intersec_posi_y2)
    if ll1 >= 12:
        y_up.append(intersec_posi_y1)
        y_down.append(intersec_posi_y2)
        x_lengh.append(x_min)
        plt.plot([x_min, x_min],[intersec_posi_y1,intersec_posi_y2])
        break
y_point= scale_y[np.where(scale_x==min(scale_x))[0][0]]  

dya = np.array(scale_y) - y_point

intersec_x = []
intersec_y = []
for ii in range(len(dya) - 1):
    mult = dya[ii]*dya[ii + 1]
    if mult < 0:
        intersec_x.append(scale_x[ii])
        intersec_x.append(scale_x[ii + 1])
        intersec_y.append(scale_y[ii])
        intersec_y.append(scale_y[ii + 1])

slope_intersec = []
const_intersec = []

for ii in range(len(intersec_x)):
    if ii % 2 != 0:
        temp = (intersec_y[ii]-intersec_y[ii - 1])/(intersec_x[ii] - intersec_x[ii - 1])
        slope_intersec.append(temp)
        temp1 = intersec_y[ii] - temp*intersec_x[ii]
        const_intersec.append(temp1)
x_point_right=(y_point-const_intersec)/slope_intersec

temp = arr_ew(ll1, len_build_total)

num_WE=np.floor((max(scale_x)-min(scale_x))/min(len_build_total))
for ii in np.arange(1,num_WE-2):
   
    for jj in range(10):
        x_min=x_min+len_build_total[jj]

        if x_min< scale_x[np.where(scale_y==max(scale_y))[0][0]]-min(len_build_total):

        # temp = arr_ew(x_point_right-x_min, len_build_total)
            intersec_posi_y1, intersec_posi_y2 = calculation_SN1(scale_x, scale_y, x_min)
            intersec_posi_y3, intersec_posi_y4 = calculation_SN1(scale_x, scale_y, x_min+min(len_build_total))

            # if intersec_posi_y1 < intersec_posi_y3:

            if  abs( intersec_posi_y3-intersec_posi_y1 ) < 12:
                x_min=x_min - len_build_total[jj]
            


        # if abs( intersec_posi_y3-intersec_posi_y1 ) > 12:

            if x_min <=max(scale_x):
                
                ll1 = np.abs(intersec_posi_y1 - intersec_posi_y2)
                if ll1 >= 12:
                    y_up.append(intersec_posi_y1)
                    y_down.append(intersec_posi_y2)
                    x_lengh.append(x_min)
                    # if jj !=0:
                    #     x_lengh.append(x_min-len_build_total[jj-1])
                    # else:
                    #     x_lengh.append(len_build_total[jj])
                    plt.plot([x_min, x_min],[intersec_posi_y1,intersec_posi_y2])
                    # plt.plot([x_min+min(len_build_total), x_min+min(len_build_total)],[intersec_posi_y3,intersec_posi_y4])
                    
    
                break

        elif x_min< scale_x[np.where(scale_y==max(scale_y))[0][0]] < x_min+min(len_build_total):

            intersec_posi_y1, intersec_posi_y2 = calculation_SN1(scale_x, scale_y, x_min)
            intersec_posi_y3, intersec_posi_y4 = calculation_SN1(scale_x, scale_y, x_min+min(len_build_total))

            if  abs( intersec_posi_y3-intersec_posi_y1 ) > 12:
                if x_min <=max(scale_x):
                
                    ll1 = np.abs(intersec_posi_y1 - intersec_posi_y2)
                    if ll1 >= 12:
                        y_up.append(intersec_posi_y1)
                        y_down.append(intersec_posi_y2)
                        x_lengh.append(x_min)
                        
                        # if jj !=0:
                        #     x_lengh.append(x_min-len_build_total[jj-1])
                        # else:
                        #     x_lengh.append(len_build_total[jj])
                        plt.plot([x_min, x_min],[intersec_posi_y1,intersec_posi_y2])
                        # plt.plot([x_min+min(len_build_total), x_min+min(len_build_total)],[intersec_posi_y3,intersec_posi_y4])
            
            elif abs( intersec_posi_y3-intersec_posi_y1 ) < 12:
                if x_min <=max(scale_x):
                
                    ll1 = np.abs(intersec_posi_y1 - intersec_posi_y2)
                    if ll1 >= 12:
                        y_up.append(intersec_posi_y1)
                        y_down.append(intersec_posi_y2)
                        x_lengh.append(x_min+10)
                        x_min=x_min+10
                        no=ii
                        # if jj !=0:
                        #     x_lengh.append(x_min-len_build_total[jj-1])
                        # else:
                        #     x_lengh.append(len_build_total[jj])
                        plt.plot([x_min, x_min],[intersec_posi_y1,intersec_posi_y2])
                        # plt.plot([x_min+min(len_build_total), x_min+min(len_build_total)],[intersec_posi_y3,intersec_posi_y4])
        
        elif x_min> scale_x[np.where(scale_y==max(scale_y))[0][0]]-min(len_build_total):

            intersec_posi_y1, intersec_posi_y2 = calculation_SN1(scale_x, scale_y, x_min)
            intersec_posi_y3, intersec_posi_y4 = calculation_SN1(scale_x, scale_y, x_min+min(len_build_total))

            if  abs( intersec_posi_y3-intersec_posi_y1 ) < 12:
                x_min=x_min - len_build_total[jj]

            if x_min <=max(scale_x):
                
                ll1 = np.abs(intersec_posi_y1 - intersec_posi_y2)
                if ll1 >= 12:
                    y_up.append(intersec_posi_y1)
                    y_down.append(intersec_posi_y2)
                    x_lengh.append(x_min)
                    # if jj !=0:
                    #     x_lengh.append(x_min-len_build_total[jj-1])
                    # else:
                    #     x_lengh.append(len_build_total[jj])
                    plt.plot([x_min, x_min],[intersec_posi_y1,intersec_posi_y2])

                    if x_min+min(len_build_total) < max(scale_x):

                        plt.plot([x_min+min(len_build_total), x_min+min(len_build_total)],[intersec_posi_y3,intersec_posi_y4])
                        x_lengh.append(x_min+min(len_build_total))
                        y_up.append(intersec_posi_y3)
                        y_down.append(intersec_posi_y4)
    
                break


print(y_up)
print(x_lengh)

for ii in range(len(y_up)-1):
    if ii == 0:
        start_point_x = x_lengh[0]
        # x_1=start_point_x
        second_point_x = x_lengh[1]
        x_build = np.hstack((start_point_x, second_point_x))
        x_build = np.hstack((x_build, np.flipud(x_build)))
        x_build = np.append(x_build, start_point_x)
        y_build = np.array([y_up[0], y_up[0], y_up[0]-12, y_up[0]-12, y_up[0]])
        plt.plot(x_build,y_build)

    elif ii == no-1:
        if y_up[ii] < y_up[ii+1]:
            y_max = y_up[ii]
            if y_down[ii] < y_down[ii+1]:
                y_min= y_down[ii+1]
                len_NS= y_max - y_min 
            elif y_down[ii] > y_down[ii+1]:
                y_min= y_down[ii]
                len_NS= y_max - y_min 



        elif y_up[ii] > y_up[ii+1]:
            y_max = y_up[ii+1]
            if y_down[ii] < y_down[ii+1]:
                y_min= y_down[ii+1]
                len_NS= y_max - y_min 
            elif y_down[ii] > y_down[ii+1]:
                y_min= y_down[ii]
                len_NS= y_max - y_min 

        build_num_NS=np.floor(len_NS/92)
        mod_NS=len_NS-(92*build_num_NS)-12
        if mod_NS > 12:
            for jj in range(int(build_num_NS)+2):
                if jj != range(int(build_num_NS)+2)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_max-jj*92, y_max-jj*92, y_max-12-jj*92, y_max-12-jj*92, y_max-jj*92])
                    plt.plot(x_build,y_build)
                elif jj == range(int(build_num_NS)+2)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_min+12, y_min+12, y_min, y_min, y_min+12])
                    plt.plot(x_build,y_build)

        else:
            for jj in range(int(build_num_NS)+1):
                if jj != range(int(build_num_NS)+1)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_max-jj*92, y_max-jj*92, y_max-12-jj*92, y_max-12-jj*92, y_max-jj*92])
                    plt.plot(x_build,y_build)
                elif jj == range(int(build_num_NS)+1)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_min+12, y_min+12, y_min, y_min, y_min+12])
                    plt.plot(x_build,y_build)

    else :
        if y_up[ii] < y_up[ii+1]:
            y_max = y_up[ii]
            if y_down[ii] < y_down[ii+1]:
                y_min= y_down[ii+1]
                len_NS= y_max - y_min 
            elif y_down[ii] > y_down[ii+1]:
                y_min= y_down[ii]
                len_NS= y_max - y_min 



        elif y_up[ii] > y_up[ii+1]:
            y_max = y_up[ii+1]
            if y_down[ii] < y_down[ii+1]:
                y_min= y_down[ii+1]
                len_NS= y_max - y_min 
            elif y_down[ii] > y_down[ii+1]:
                y_min= y_down[ii]
                len_NS= y_max - y_min 

        build_num_NS=np.floor(len_NS/92)
        mod_NS=len_NS-(92*build_num_NS)-12
        if mod_NS > 12:
            for jj in range(int(build_num_NS)+2):
                if jj != range(int(build_num_NS)+2)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_max-jj*92, y_max-jj*92, y_max-12-jj*92, y_max-12-jj*92, y_max-jj*92])
                    plt.plot(x_build,y_build)
                elif jj == range(int(build_num_NS)+2)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_min+12, y_min+12, y_min, y_min, y_min+12])
                    plt.plot(x_build,y_build)

        else:
            for jj in range(int(build_num_NS)+1):
                if jj != range(int(build_num_NS)+1)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_max-jj*92, y_max-jj*92, y_max-12-jj*92, y_max-12-jj*92, y_max-jj*92])
                    plt.plot(x_build,y_build)
                elif jj == range(int(build_num_NS)+1)[-1]:
                    start_point_x = x_lengh[ii]
                    # x_1=start_point_x
                    second_point_x = x_lengh[ii+1]
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([y_min+12, y_min+12, y_min, y_min, y_min+12])
                    plt.plot(x_build,y_build)







plt.show()
# %%
