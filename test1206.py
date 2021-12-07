# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely.geometry as geometry
from io import StringIO


MIN_LENGTH = 18
MAX_LENGTH = 50
area = 40000
plot_Ratio = 2.0
num_tree = np.ceil(area / 300)
tree_distance = 15
area = 100000
ratio = 2

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
        x1 = data[i % num][0] - data[(i - 1) % num][0]
        y1 = data[i % num][1] - data[(i - 1) % num][1]
        x2 = data[(i + 1) % num][0] - data[i % num][0]
        y2 = data[(i + 1) % num][1] - data[i % num][1]

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



def slope_cal(xx, yy):
    """求边的斜率"""
    slope_list = []
    for ii in range(len(xx)-1):
        temp=(yy[ii+1]-yy[ii])/(xx[ii+1]-yy[ii])
        slope_list.append(temp)
    return slope_list

def function_line(xx,yy):
    """求直线方程"""
    A = []
    B = []
    C = []
    n = len(xx)
    for ii in range(n-1):
        A.append(yy[ii+1] - yy[ii])
        B.append(xx[ii] - xx[ii+1])
        C.append(xx[ii+1] * yy[ii] - xx[ii] * yy[ii+1])
    return A, B, C

def coor_y(xx, yy, x):
    """ x=x 竖线 与边交点的纵坐标"""
    A, B, C = function_line(xx, yy)
    dya = x - xx
    n = len(dya)
    coo_y=[]
    for i in range(n-1):
        if dya[i] * dya[i+1] < 0:
            coo_y.append((-A[i] * x - C[i]) / B[i])
    return coo_y

def coor_x(xx, yy, y):
    """y =y 横线 与边交点的横坐标"""
    A, B, C = function_line(xx, yy)
    dya = y - yy
    n = len(dya)
    coo_x = []
    for i in range(n - 1):
        if dya[i] * dya[i + 1] < 0:
            coo_x.append((-B[i] * y - C[i]) / A[i])
    return coo_x

def length_combination(len_building=None):
    # Calculation length combination
    if len_building is None:
        len_building = [7, 9, 11, 15]
    res = set()
    for i in len_building:
        for j in len_building:
            if MIN_LENGTH < i + j < MAX_LENGTH:
                res.add(i + j)
            for k in len_building:
                if MIN_LENGTH < i + j + k < MAX_LENGTH:
                    res.add(i + j + k)
                for l in len_building:
                    if MIN_LENGTH < i + j + k + l < MAX_LENGTH:
                        res.add(i + j + k + l)
    return sorted(res)

def cal_ave_verlin(xx, len_build_list):
    """计算均分时竖线的排楼方法"""
    x_max = max(xx)
    x_min = min(xx)

    mod_lin = []
    num_lin = []
    len_buil = []
    n = len(len_build_list)
    for ii in range(n):
        if ii == 0:
            temp = np.floor((x_max - x_min + 15 * 2) / (len_build_list[ii]) + 15)
            num_lin.append(temp)
            mod_lin.append((x_max - x_min + 15 * 2) % (len_build_list[ii] + 15))
            len_buil = len_build_list[ii]
        else:
            if (x_max - x_min + 15 * 2) % (len_build_list[ii] + 15) < mod_lin[0]:
                mod_lin[0] = (x_max - x_min + 15 * 2) % (len_build_list[ii] + 15)
                temp = np.floor((x_max - x_min + 15 * 2) / (len_build_list[ii]) + 15)
                num_lin[0] = temp
                len_buil = len_build_list[ii]
    return num_lin, len_buil



def plot_vertical_line(xx, yy, len_build_list, len_buil):
    """画竖线"""
    x_ver_lin = []
    y_up = []
    y_down = []
    # while x_ver_lin[-1] < max(xx) - min(len_build_list):
    temp = np.linspace(0.2, 21, 50)
    x_min = min(xx)
    for ii in temp:
        x_min1 = x_min + ii
        y = coor_y(xx, yy, x_min1)

        if abs(y[0] - y[1]) >= 12:
            x_ver_lin.append((x_min1))
            y_up.append((max(y)))
            y_down.append(min(y))
            plt.plot([x_min1, x_min1], [y[-1], y[0]], color='grey', linestyle='--', linewidth=0.5)
            break
    x_min= x_ver_lin[-1]
    while x_min <= max(xx):
        x_min += len_buil + 15
        if x_min > max(xx):
            break
        else:
            y = coor_y(xx, yy, x_min)
            y_up.append((max(y)))
            y_down.append(min(y))
            x_ver_lin.append((x_min))
            plt.plot([x_min, x_min], [y[-1], y[0]], color='grey', linestyle='--', linewidth=0.5)
    return  x_ver_lin, y_up, y_down


def plot_rectangle(xx, yy, x_ver_lin, y_up, len_buil):
    """画楼（矩形）"""

    x_fir = []
    y_fir = []
    len_bui =[]

    n=len(x_ver_lin)
    for ii in range(n):
        y_up1 = y_up[ii]
        y_down1 = y_down[ii]
        if ii == 0:
            y = coor_y(xx, yy, x_ver_lin[ii] + len_buil)
            start_point_x = x_ver_lin[ii]
            second_point_x = start_point_x + len_buil
            x_build = np.hstack((start_point_x, second_point_x))
            x_build = np.hstack((x_build, np.flipud(x_build)))
            x_build = np.append(x_build, start_point_x)
            y_build = np.array([min([max(y), y_up1]), min([max(y), y_up1]),
            min([max(y), y_up1])-12, min([max(y), y_up1])-12, 
            min([max(y), y_up1])])
            x_fir.append(start_point_x)
            y_fir.append(min([max(y), y_up1]))
            len_bui.append(len_buil)
            # plt.plot(x_build, y_build)

            y = coor_y(xx, yy, second_point_x)
            y_min_lin = min(y)
            y_down_re = min([max(y), y_up1])-12

            while y_down_re > y_min_lin + 80 * 2:
                y_up_re = y_down_re -80
                y_down_re = y_down_re - 92
                x1 = min(coor_x(xx, yy, y_up_re))
                x2 = min(coor_x(xx, yy,y_down_re))
                x_left_re = max([x1, x2])
                len_dis = x_ver_lin[ii + 1] - x_left_re
                if len_dis >= min(len_build_list):
                    modd = len_dis - np.array([len_build_list])
                    nn=np.shape(modd)[1]
                    for jj in range(nn - 1):
                        if modd[0,jj] > 0 and modd[0,jj + 1] < 0:
                            len_build = len_build_list[jj]
                            start_point_x = x_left_re
                            second_point_x = start_point_x + len_build
                            x_build = np.hstack((start_point_x, second_point_x))
                            x_build = np.hstack((x_build, np.flipud(x_build)))
                            x_build = np.append(x_build, start_point_x)
                            y_build = np.array([y_up_re, y_up_re,y_down_re, y_down_re, y_up_re])
      
                    x_fir.append(start_point_x)
                    y_fir.append(y_up_re)
                    len_bui.append(len_build)
                    # plt.plot(x_build, y_build)

            while y_down_re > y_min_lin + 80 and y_down_re < y_min_lin + 80 * 2:
                y_up_re = y_down_re -80
                y_down_re = y_down_re - 92
                x1 = min(coor_x(xx, yy, y_up_re))
                x2 = min(coor_x(xx, yy,y_down_re))
                x_left_re = max([x1, x2])
                len_dis = x_ver_lin[ii + 1] - x_left_re
                if len_dis >= min(len_build_list):
                    modd = len_dis - np.array([len_build_list])
                    nn=np.shape(modd)[1]
                    for jj in range(nn - 1):
                        if modd[0,jj] > 0 and modd[0,jj + 1] < 0:
                            len_build = len_build_list[jj]
                            start_point_x = x_left_re
                            second_point_x = start_point_x + len_build
                            x_build = np.hstack((start_point_x, second_point_x))
                            x_build = np.hstack((x_build, np.flipud(x_build)))
                            x_build = np.append(x_build, start_point_x)
                            y_build = np.array([y_up_re, y_up_re,y_down_re, y_down_re, y_up_re])
                    x_fir.append(start_point_x)
                    y_fir.append(y_up_re)
                    len_bui.append(len_build)
                    # plt.plot(x_build, y_build)
                            

        elif ii == n - 1:
            if x_ver_lin[ii] + min(len_build_list) < max(xx):
                y = coor_y(xx, yy, x_ver_lin[ii] + min(len_build_list))
                if max(y) - min(y) >= 12:
                    start_point_x = x_ver_lin[ii]
                    second_point_x = start_point_x + len_buil
                    x_build = np.hstack((start_point_x, second_point_x))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, start_point_x)
                    y_build = np.array([min([max(y), y_up1]), min([max(y), y_up1]),
                    min([max(y), y_up1])-12, min([max(y), y_up1])-12, 
                    min([max(y), y_up1])])
                    x_fir.append(start_point_x)
                    y_fir.append(min([max(y), y_up1]))
                    len_bui.append(len_buil)
                    # plt.plot(x_build, y_build)

        else:
            y = coor_y(xx, yy, x_ver_lin[ii] + len_buil)
            max_y = min(y_up1, max(y))
            min_y = max(y_down1, min(y))
            len_NS = max_y - min_y

            build_num_NS=np.floor(len_NS/92)
            mod_NS=len_NS-(92*build_num_NS)-12
            if mod_NS > 12:
                for jj in range(int(build_num_NS)+2):
                    if jj != range(int(build_num_NS)+2)[-1] and jj != range(int(build_num_NS)+2)[-2] and jj != 0:
                        x_left_re = x_ver_lin[ii]
                        len_dis = x_ver_lin[ii + 1] - x_left_re
                        if len_dis >= min(len_build_list):
                            modd = len_dis - np.array([len_build_list])
                            nn=np.shape(modd)[1]
                            for kk in range(nn - 1):
                                if modd[0,kk] >= 0 and modd[0,kk + 1] <= 0:
                                    len_build = len_build_list[kk]
                                    start_point_x = x_ver_lin[ii]
                                    second_point_x = start_point_x + len_build
                                    x_build = np.hstack((start_point_x, second_point_x))
                                    x_build = np.hstack((x_build, np.flipud(x_build)))
                                    x_build = np.append(x_build, start_point_x)
                                    y_build = np.array([max_y-jj*92, max_y-jj*92, max_y-12-jj*92, max_y-12-jj*92, max_y-jj*92])
                            
                            x_fir.append(start_point_x)
                            y_fir.append(y_build[0])
                            len_bui.append(len_build)
                            # plt.plot(x_build,y_build)


                    elif jj == 0:
                        start_point_x = x_ver_lin[ii]
                        second_point_x = start_point_x + len_buil
                        x_build = np.hstack((start_point_x, second_point_x))
                        x_build = np.hstack((x_build, np.flipud(x_build)))
                        x_build = np.append(x_build, start_point_x)
                        y_build = np.array([max_y-jj*92, max_y-jj*92, max_y-12-jj*92, max_y-12-jj*92, max_y-jj*92])
                        x_fir.append(start_point_x)
                        y_fir.append(y_build[0])
                        len_bui.append(len_buil)
                        # plt.plot(x_build,y_build)

                    elif jj == range(int(build_num_NS)+2)[-1]:
                        start_point_x = x_ver_lin[ii]
                        second_point_x = start_point_x + len_buil
                        x_build = np.hstack((start_point_x, second_point_x))
                        x_build = np.hstack((x_build, np.flipud(x_build)))
                        x_build = np.append(x_build, start_point_x)
                        y_build = np.array([min_y+12, min_y+12, min_y, min_y, min_y+12])
                        x_fir.append(start_point_x)
                        y_fir.append(y_build[0])
                        len_bui.append(len_buil)
                        # plt.plot(x_build,y_build)

                    elif jj == range(int(build_num_NS)+2)[-2]:
                        x_left_re = x_ver_lin[ii]
                        len_dis = x_ver_lin[ii + 1] - x_left_re
                        if len_dis >= min(len_build_list):
                            modd = len_dis - np.array([len_build_list])
                            nn=np.shape(modd)[1]
                            for kk in range(nn - 1):
                                if modd[0,kk] >= 0 and modd[0,kk + 1] <= 0:
                                    len_build = len_build_list[kk]
                                    start_point_x = x_ver_lin[ii]
                                    second_point_x = start_point_x + len_build
                                    x_build = np.hstack((start_point_x, second_point_x))
                                    x_build = np.hstack((x_build, np.flipud(x_build)))
                                    x_build = np.append(x_build, start_point_x)
                                    y_build = np.array([(max_y-(jj-1)*92-min_y)/2+min_y, (max_y-(jj-1)*92-min_y)/2+min_y, 
                                    (max_y-(jj-1)*92-min_y)/2+min_y-12, (max_y-(jj-1)*92-min_y)/2+min_y-12, 
                                    (max_y-(jj-1)*92-min_y)/2+min_y])
                                    
                              
                            x_fir.append(start_point_x)
                            y_fir.append(y_build[0])
                            len_bui.append(len_build)
                            # plt.plot(x_build,y_build)

            else:
                for jj in range(int(build_num_NS)+1):
                    if jj != range(int(build_num_NS)+2)[-1] and jj != range(int(build_num_NS)+2)[-2] and jj != 0:
                        x_left_re = x_ver_lin[ii]
                        len_dis = x_ver_lin[ii + 1] - x_left_re
                        if len_dis >= min(len_build_list):
                            modd = len_dis - np.array([len_build_list])
                            nn=np.shape(modd)[1]
                            for kk in range(nn - 1):
                                if modd[0,kk] >= 0 and modd[0,kk + 1] <= 0:
                                    len_build = len_build_list[kk]
                                    start_point_x = x_ver_lin[ii]
                                    second_point_x = start_point_x + len_build
                                    x_build = np.hstack((start_point_x, second_point_x))
                                    x_build = np.hstack((x_build, np.flipud(x_build)))
                                    x_build = np.append(x_build, start_point_x)
                                    y_build = np.array([max_y-jj*92, max_y-jj*92, max_y-12-jj*92, max_y-12-jj*92, max_y-jj*92])
                                    
                            x_fir.append(start_point_x)
                            y_fir.append(y_build[0])
                            len_bui.append(len_build)
                            # plt.plot(x_build,y_build)

                    elif jj == 0:
                        start_point_x = x_ver_lin[ii]
                        second_point_x = start_point_x + len_buil
                        x_build = np.hstack((start_point_x, second_point_x))
                        x_build = np.hstack((x_build, np.flipud(x_build)))
                        x_build = np.append(x_build, start_point_x)
                        y_build = np.array([max_y-jj*92, max_y-jj*92, max_y-12-jj*92, max_y-12-jj*92, max_y-jj*92])
                        x_fir.append(start_point_x)
                        y_fir.append(y_build[0])
                        len_bui.append(len_buil)
                        # plt.plot(x_build,y_build)

                    elif jj == range(int(build_num_NS)+1)[-1]:
                        start_point_x = x_ver_lin[ii]
                        # x_1=start_point_x
                        second_point_x = start_point_x + len_buil
                        x_build = np.hstack((start_point_x, second_point_x))
                        x_build = np.hstack((x_build, np.flipud(x_build)))
                        x_build = np.append(x_build, start_point_x)
                        y_build = np.array([min_y+12, min_y+12, min_y, min_y, min_y+12])
                        x_fir.append(start_point_x)
                        y_fir.append(y_build[0])
                        len_bui.append(len_buil)
                        # plt.plot(x_build,y_build)

                    elif jj == range(int(build_num_NS)+1)[-2]:
                        x_left_re = x_ver_lin[ii]
                        len_dis = x_ver_lin[ii + 1] - x_left_re
                        if len_dis >= min(len_build_list):
                            modd = len_dis - np.array([len_build_list])
                            nn=np.shape(modd)[1]
                            for kk in range(nn - 1):
                                if modd[0,kk] >= 0 and modd[0,kk + 1] <= 0:
                                    len_build = len_build_list[kk]
                                    start_point_x = x_ver_lin[ii]
                                    # x_1=start_point_x
                                    second_point_x = start_point_x + len_build
                                    x_build = np.hstack((start_point_x, second_point_x))
                                    x_build = np.hstack((x_build, np.flipud(x_build)))
                                    x_build = np.append(x_build, start_point_x)
                                    y_build = np.array([(max_y-(jj+1)*92-min_y)/2, (max_y-(jj+1)*92-min_y)/2, 
                                    (max_y-(jj+1)*92-min_y)/2-12, (max_y-(jj+1)*92-min_y)/2-12, 
                                    (max_y-(jj+1)*92-min)/2])

                            x_fir.append(start_point_x)
                            y_fir.append(y_build[0])
                            len_bui.append(len_build)
                            # plt.plot(x_build,y_build)

    return x_fir, y_fir, len_bui

def get_info_build(x_ver_lin, x_fir, y_fir, len_bui):
    m=len(x_ver_lin)
    n=len(x_fir)
    info_build=[]
    info_build_list = []
    for ii in range(m-1):
        n_initial = 0
        info_build1=[]
        for jj in range(n):
            if (x_ver_lin[ii + 1] > x_fir[jj]) * ( x_fir[jj] >= x_ver_lin[ii]):
                # print(x_ver_lin[ii + 1],x_fir[jj],x_ver_lin[ii], jj)
                if n_initial == 0:
                    info_build=[x_fir[jj], y_fir[jj], len_bui[jj], 92]
                else:
                    info_build=[x_fir[jj], y_fir[jj], len_bui[jj], abs(y_fir[jj]-y_fir[jj-1])]
                info_build1.append(info_build)
                n_initial += 1
        info_build_list.append(info_build1)
    return info_build_list

def get_build_ratio(area, info_build_list):
    temp1 = 0
    for list1 in info_build_list:
        for list2 in list1:
            num_floor=np.floor(list2[3]/3.3)
            temp1 += list2[2]*12*num_floor * 0.8
    build_ratio  = temp1 / area
    return build_ratio

def replot_build(area, info_build_list):
    while get_build_ratio(area, info_build_list) > 2:
        for list1 in info_build_list:
            if len(list1) > 2:
                del list1[1]

    for list1 in info_build_list:
        n=np.shape(list1)[0]
        len_fir_end=list1[0][1]-list1[-1][1]
        y_change= len_fir_end / (n-1)
        for ii in range(n):
            if ii != 0 or ii != n-1:
                list1[ii][1]=list1[0][1]-ii*y_change

    y_fir = []
    x_fir = []
    len_bui = []
    for list1 in info_build_list:
        for list2 in list1:
            x=[list2[0], list2[0] + list2[2], list2[0] + list2[2], list2[0], list2[0]]
            y=[list2[1], list2[1], list2[1] - 12,list2[1] - 12, list2[1]]
            y_fir.append(list2[1])
            x_fir.append(list2[0])
            len_bui.append(list2[2])
            plt.plot(x,y)

    return info_build_list, get_build_ratio(area, info_build_list), x_fir, y_fir,len_bui
            
            
def polyg(xx,yy):
    """
    对给定坐标形成区域
    """
    aa=[]
    for ii in range(len(xx)-1):
        aa.append(xx[ii])
        aa.append(yy[ii])
        f1=""
        for ii in range(len(aa)):
            if ii%2 ==0:
                
                f1 += str(aa[ii]) + " "
            else:
                f1 += str(aa[ii]) + "\n"
    return f1


def slope_cal(xx, yy):
    """求边的斜率"""
    slope_list = []
    for ii in range(len(xx)-1):
        temp=(yy[ii+1]-yy[ii])/(xx[ii+1]-yy[ii])
        slope_list.append(temp)
    return slope_list


def cal_len_edge(x1,y1,x2,y2):
    """求两点的长度"""
    return np.sqrt((y2-y1)**2+(x2-x1)**2)


def points(xx, yy,ff):
    """
    判断 ff 是否在 xx, yy 形成的区域内 
    """
    pg = Polygon(np.genfromtxt(StringIO(ff))) # 生成Polygon实例
    return geometry.Point([xx, yy]).within(pg)

def plot_tree_edge(xx,yy):
    """
    : xx: 边的横坐标
    : yy: 边的纵坐标
    : tree_distance: 树之间的距离
    """
    x_tree_edge=[]
    y_tree_edge=[]
    r_tree_edge=[]
    for ii in range(len(xx)-1):
        ll=cal_len_edge(xx[ii],yy[ii],xx[ii+1],yy[ii+1])
        num_tree=np.floor(ll/tree_distance)
        mod_dis=np.mod(ll,tree_distance)
        if mod_dis <= tree_distance/2:
            tree_dis= ll/ num_tree
            ll1=0
            x=xx[ii]
            y=yy[ii]
            while ll1<=ll-mod_dis:
                x_tree_edge.append(x)
                y_tree_edge.append(y)
                ll1 += tree_dis
                r=random.uniform(1.5,3)
                r_tree_edge.append(r)
                theta = np.arange(0, 2*np.pi, 0.01)
                x1 = x + r * np.cos(theta)
                y1 = y + r * np.sin(theta)
                plt.plot(x1,y1, color='g')

                x += 15* (xx[ii+1]-xx[ii])/ll
                y += 15* (yy[ii+1]-yy[ii])/ll

        else:
            ll1=0
            x=xx[ii]
            y=yy[ii]
            while ll1<=ll:
                ll1 += 15
                r=random.uniform(1.5,3)
                x_tree_edge.append(x)
                y_tree_edge.append(y)
                r_tree_edge.append(r)
                theta = np.arange(0, 2*np.pi, 0.01)
                x1 = x + r * np.cos(theta)
                y1 = y+ r * np.sin(theta)
                plt.plot(x1,y1, color='g')

                x += 15* (xx[ii+1]-xx[ii])/ll
                y += 15* (yy[ii+1]-yy[ii])/ll
    
    return x_tree_edge, y_tree_edge, r_tree_edge

def plot_tree_region(xx,yy):
    """
    画区域内的树
    : xx: 退线之后点的横坐标
    : yy: 退线之后点的纵坐标
    : num_tree: 要种树的量
    """
    x_max=max(xx)
    x_min=min(xx)
    y_max=max(yy)
    y_min=min(yy)

    x_tree_region=[]
    y_tree_region=[]
    r_tree_region=[]
    while  len(y_tree_region) <= num_tree:
        x2=random.uniform(x_min,x_max)
        y2=random.uniform(y_min,y_max)
        if points(x2,y2, polyg(xx,yy)) == True :

            temp=[]
            for ll in range(len(len_bui)):
                xx_re=np.array([x_fir[ll]-15, x_fir[ll]+len_bui[ll]+15, x_fir[ll]+len_bui[ll]+15, x_fir[ll]-15, x_fir[ll]-15])
                yy_re=[y_fir[ll]+15, y_fir[ll]+15, y_fir[ll]-12-15, y_fir[ll]-12-15, y_fir[ll]-12-15]
                
                temp.append(points(x2,y2, polyg(xx_re, yy_re)))
            if sum(temp) == 0:
                y_tree_region.append(y2)
                x_tree_region.append(x2)
                r=random.uniform(1.5,3)
                r_tree_region.append(r)
                theta = np.arange(0, 2*np.pi, 0.01)
                x1 = x2 + r * np.cos(theta)
                y1 = y2+ r * np.sin(theta)
                plt.plot(x1,y1, color='g')

    return x_tree_region, y_tree_region, r_tree_region
                            
                        
                            


if __name__ == '__main__':
    # xx =  np.array([0, 150, 250, 350, 200, 110, 0, 0])
    # yy =  np.array([340, 440, 360, -20, -70, -120, 80, 340])
    xx = np.array([0, 150, 270, 250, 200, 110, 30, 0]) 
    yy = np.array([340, 460, 390, -20, -70, -120, 80, 340]) 
    plt.plot(xx, yy)
    xx, yy = delVertex(xx, yy, threshold=0.005)
    plt.plot(xx, yy)
    x_scale, y_scale = scale(xx, yy, sec_dis=10)
    plt.plot(x_scale, y_scale)

    len_build_list = length_combination()
    num_lin, len_buil = cal_ave_verlin(x_scale, len_build_list)
    x_ver_lin, y_up, y_down = plot_vertical_line(x_scale, y_scale, len_build_list, len_buil)
    print(len_buil)
    x_fir, y_fir, len_bui = plot_rectangle(x_scale, y_scale, x_ver_lin, y_up, len_buil)
    info_build_list = get_info_build(x_ver_lin, x_fir, y_fir, len_bui)
    info_build_list, build_ratio, x_fir, y_fir, len_bui = replot_build(area,info_build_list)
    x_tree_edge, y_tree_edge, r_tree_edge = plot_tree_edge(xx,yy)
    x_tree_region, y_tree_region, r_tree_region = plot_tree_region(x_scale,y_scale)
    
    print(build_ratio)


    plt.show()
# %%
