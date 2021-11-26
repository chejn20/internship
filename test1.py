# %%

import numpy as np
import matplotlib.pyplot as plt
xx=np.array([0,150,250,350, 200,110, 10,0])
yy=np.array([340,440,360,-20,-70 ,-120, 80,340])
# yy = np.array([300, 300, 330, 10, 20, 300])
plt.figure(dpi=300,figsize=(4,4))
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
max_x=np.max(xx)
max_y=np.max(yy)
min_x=np.min(xx)
min_y=np.min(yy)
mid_x=(max_x+min_x)/2
mid_y=(max_y+min_y)/2


xx1=[]
yy1=[]
for ii in range(len(xx )):
    r=np.sqrt((xx[ii]-mid_x)**2+(yy[ii]-mid_y)**2)
    x=(xx[ii]-mid_x)*np.cos(np.pi/2)-(yy[ii]-mid_y)*np.sin(np.pi/2)+mid_x
    y=(xx[ii]-mid_x)*np.sin(np.pi/2)-(yy[ii]-mid_y)*np.cos(np.pi/2)+mid_y
    xx1.append(x)
    yy1.append(y)

xx1=np.array(xx1)
yy1=np.array(yy1)
plt.plot(xx1,yy1)
# plt.show()

# %%

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

plt.plot(xx1, yy1)
xx, yy = delVertex(xx1, yy1, threshold = 0.005)
plt.plot(xx, yy)
scale_x, scale_y = scale(xx, yy, sec_dis = 10)
plt.plot(scale_x, scale_y)
# plt.show()

# %%  所有楼高组合

high_build_total=np.arange(21,80,3)

# %%
def closest(mylist, Number):
    # Results of two straight lengths
  """
    :param mylist: Unfilled building length
    :param Number: Remainder of the ground after the row
    :return: Row length and remaining land
    """
  answer = Number - mylist
  if all(answer < 0) == True:
        return [0, Number]
  else:
        answer1 = answer[(answer >= 0)]
        answer1 = min(answer1)
        indd = np.where(answer == answer1)
        return [mylist[indd[0]], answer1]

# ll=325  # as example
def arr_ew1(ll, ll_build):
    # ll is the East-West length, ll_ Build is the length combination of all buildings
    sche = {}

    for ii in np.flipud(ll_build):
        times = int((ii - ll_build[0])/3)
        build_num=np.floor(ll/ii)
        mod_num = np.mod(ll, ii)
        if mod_num < min(ll_build):
                sche_name = str(ii)
                sche[sche_name] = [ii, build_num , mod_num]
        else:
            aa = closest(ll_build[:times], mod_num)
            sche_name = str(ii)
            sche[sche_name] = [ii, build_num , mod_num, aa[0], aa[1]]


    # for ii in np.flipud(ll_build):
    #     times = ii - ll_build[0]
    #     if ll > (ii + 2.5)*2:
    #         ll1 = ll - (ii + 2.5)*2
    #         build_num = np.floor(ll1/(ii + 5))
    #         mod_num = np.mod(ll1, (ii + 5))
    #         if mod_num < min(ll_build):
    #             sche_name = str(ii)
    #             sche[sche_name] = [ii, build_num + 2, mod_num, 2*2.5 + build_num*5 + mod_num]
    #         else:
    #             aa = closest(ll_build[:times - 1], mod_num)
    #             sche_name = str(ii)
    #             sche[sche_name] = [ii, build_num + 2, mod_num, aa[0], aa[1], 2*2.5 + (build_num+1)*5 + aa[1]]

    #     if ll < (ii + 2.5)*2  and  ll > ii + 5:
    #         ll1 = ll - ii
    #         build_num = np.floor(ll1/(ii + 5))
    #         mod_num = np.mod(ll1, (ii + 5))
    #         if mod_num < min(ll_build):
    #             sche_name = str(ii)
    #             sche[sche_name] = [ii, build_num + 1, mod_num, 2*2.5 + build_num*5 + mod_num]
    #         else:
    #             aa = closest(ll_build[:times - 1], mod_num)
    #             sche_name = str(ii)
    #             sche[sche_name] = [ii,build_num + 1, mod_num, aa[0], aa[1], 2*2.5 + (build_num + 1)*5 + aa[1]]

    return sche

# %%
def calculation_SN1(xx, yy, ymax):
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
# %%
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

len_bulid_total = np.hstack((len_2build, len_3build, len_4build))
len_bulid_total = np.sort(len_bulid_total[(len_bulid_total > 18) & (len_bulid_total < 50)])
len_bulid_total = np.unique(len_bulid_total)



# %%
y_max=np.max(scale_y)

for ii in np.linspace(0.01, 40, 20000):
    y_max = y_max - ii
    intersec_posi_x1, intersec_posi_x2 = calculation_SN1(scale_x, scale_y, y_max)
    ll1 = np.abs(intersec_posi_x2 - intersec_posi_x1)
    if ll1 >= 12:
        break


# %%
insert_x=np.array([])
insert_y=np.array([])
dya=scale_x-intersec_posi_x1
for ii in range(len(scale_x)-1):
    # print(ii)
    mult = dya[ii]*dya[ii + 1]
    if mult < 0:
        insert_x=np.append(insert_x,[scale_x[ii],scale_x[ii+1]])
        insert_y=np.append(insert_y,[scale_y[ii],scale_y[ii+1]])
        print(insert_x,insert_y)

for ii in range(len(insert_y)-1):
    if insert_y[ii] < y_max and y_max < insert_y[ii+1]:
        insert_x=np.delete(insert_x,[ii,ii+1])
        insert_y=np.delete(insert_y,[ii,ii+1])
        print(insert_x,insert_y)
        break

# %%
slope=slope_cal(insert_x,insert_y)[0]
constant=insert_y[0]-slope*insert_x[0]
y_min=slope*intersec_posi_x1+constant

leng_NS=np.abs(y_max-y_min)

sche1=arr_ew1(leng_NS,len_bulid_total)

mod_aera = []
for ii in len_bulid_total:
    if len(sche1[str(ii)]) == 3:
        mod_aera.append(sche1[str(ii)][2])
    if len(sche1[str(ii)]) == 5:
        mod_aera.append(sche1[str(ii)][4])

min_mod = np.where(mod_aera == min(mod_aera))

start_point_x=intersec_posi_x1
second_point_x=intersec_posi_x1+12
x_build = np.hstack((start_point_x, second_point_x))
x_build = np.hstack((x_build, np.flipud(x_build)))
x_build = np.append(x_build, start_point_x)

y_build = np.array([y_max, y_max, y_max - sche1[str(36)][3][0], y_max - sche1[str(36)][3][0], y_max])

plt.plot(x_build,y_build)

plt.show()    



# %%
