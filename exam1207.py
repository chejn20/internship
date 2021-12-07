# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from shapely.geometry.polygon import Polygon
import shapely.geometry as geometry
from io import StringIO

coe_suntime = 1.3  # 日照率
MIN_DISTANCE = 12  # 楼宽
MIN_LENGTH = 18  # 最短楼长
MAX_LENGTH = 50  # 最长楼长
MAX_FLOOR_NUM = 24  # 最大楼高
UNIT_HIGH = 3.3  # 每层楼高
height_limit = MAX_FLOOR_NUM * UNIT_HIGH  # 限高
BUILDING_INTERVAL = height_limit * coe_suntime  # 楼间距
AREA = 98000  # 土地面积
plot_Ratio = 1.8  # 容积率
num_tree = np.ceil(AREA / 1800)  # 树的个数
tree_distance = 15  # 树间距
k = 0.8  # 容积率系数
houseTypes = [[7, 12], [9, 12], [11, 12], [15, 12]]  # 户型组合


def miller_to_XY(lon, lat):
    # lon: 经度，西经为负数
    # lat: 维度，南纬为负数
	L = 6381372 * math.pi * 2  # 地球周长

	W = L  # 平面展开后，x轴等于周长
	H = L / 2  # y轴约等于周长一半
	mill = 2.3  # 米勒投影中的一个常数，范围大约在正负2.3之间
	x = lon * math.pi / 180  # 将经度从度数转换为弧度
	y = lat * math.pi / 180  # 将纬度从度数转换为弧度

    # 米勒投影的转换:
	y = 1.25 * math.log(math.tan( 0.25 * math.pi + 0.4 * y ))

    # 将弧度转为实际距离:
	x1 = ( W / 2 ) + ( W / (2 * math.pi) ) * x
	y1 = ( H / 2 ) - ( H / ( 2 * mill ) ) * y

    # 转换结果的单位是米

	return x1, y1


def XY_to_miller(x1,y1):

    L = 6381372 * math.pi * 2  # 地球周长
    W = L  # 平面展开后，x轴等于周长
    H = L / 2  # y轴约等于周长一半
    mill = 2.3  # 米勒投影中的一个常数，范围大约在正负2.3之间

    x = x1 * math.pi * 2 / W -math.pi
    y = - (2 * y1 * mill / H - mill)

    y = (math.atan(math.e ** (y / 1.25)) - 0.25 * math.pi) / 0.4
    
    lon = x * 180 / math.pi
    lat = y * 180 / math.pi

    return lon, lat



def get_XY_coor(lon,lat):
    X=[]
    Y=[]
    n= len(lon)
    for ii in range(n):
        X.append(miller_to_XY(lon[ii],lat[ii])[0])
        Y.append(miller_to_XY(lon[ii],lat[ii])[1])
    return X, Y

def get_lonlat_coor(X, Y):
    lon=[]
    lat=[]
    n= len(X)
    for ii in range(n):
        lon.append(XY_to_miller(X[ii],Y[ii])[0])
        lat.append(XY_to_miller(X[ii],Y[ii])[1])
    return lon, lat


def remove_same(list_):
    # list去重
    res_list = []
    for x in list_:
        if x not in res_list:
            res_list.append(x)
    return res_list


def plot_ratio(floor, rectangle_area, area=AREA):
    """计算容积率"""
    return floor * rectangle_area * k / area


def full_info_of_building(full_coord_list):
    """根据坐标生成楼的完整信息
    x_left: 楼的左侧横坐标
    x_right: 楼的右侧横坐标
    y_up: 楼的北侧纵坐标
    y_down: 楼的南侧纵坐标
    rectangle_area: 楼的占地面积
    floor: 楼的层数 """

    full_info_list = []
    for coord_list in full_coord_list:
        info_list = []
        x_left, x_right, y_up, y_down = coord_list[0]
        rectangle_area = (x_right - x_left) * (y_up - y_down)
        floor = MAX_FLOOR_NUM  # 最北侧默认最大楼层
        info_list.append(coord_list[0] + [rectangle_area, floor])
        del_list = []
        if len(coord_list) > 1:
            for i in range(1, len(coord_list)):
                x_left, x_right, y_up, y_down = coord_list[i]
                rectangle_area = (x_right - x_left) * (y_up - y_down)
                floor = min(math.floor((coord_list[i - 1][3] - y_up) / UNIT_HIGH), MAX_FLOOR_NUM)
                if floor < 3:
                    del_list.append(coord_list[i])
                    continue
                info_list.append(coord_list[i] + [rectangle_area, floor])
            for x in del_list:
                coord_list.remove(x)
        full_info_list.append(info_list)
    return full_info_list


def total_plot_ratio(full_info_list):
    """计算土地总容积率"""
    res = 0
    for info_list in full_info_list:
        for info in info_list:
            _, _, _, _, rectangle_area, floor = info
            res += plot_ratio(floor, rectangle_area)

    return res


def get_build_ratio(AREA, full_info_list):
    temp1 = 0
    for list1 in full_info_list:
        for list2 in list1:
            num_floor=list2[5]
            temp1 += list2[4]*num_floor * 0.8
    build_ratio  = temp1 / AREA
    return build_ratio


def plot_building(full_coord_list):
    """根据坐标信息画楼"""
    for coord_list in full_coord_list:
        for coord in coord_list:
            x_left, x_right, y_up, y_down = coord
            # plt.plot([x_left, x_right, x_right, x_left, x_left],
            #          [y_up, y_up, y_down, y_down, y_up],
            #          color='k', linewidth=0.5)


def length_combination(len_building=None):
    # Calculation length combination
    if len_building is None:
        len_building = [7, 9, 11, 15]
    res = set()
    for i in len_building:
        for j in len_building:
            if MIN_LENGTH <= i + j <= MAX_LENGTH:
                res.add(i + j)
            for k in len_building:
                if MIN_LENGTH <= i + j + k <= MAX_LENGTH:
                    res.add(i + j + k)
                for l in len_building:
                    if MIN_LENGTH <= i + j + k + l <= MAX_LENGTH:
                        res.add(i + j + k + l)
    return sorted(res)

def get_combination(length_bulid):
    # Calculation length combination
    
    len_building = [7, 9, 11, 15]
    res = set()
    for i in len_building:
        for j in len_building:
            if  i + j == length_bulid:
                return i, j
                
            for k in len_building:
                if  i + j + k == length_bulid:
                    return i, j, k
                    
                for l in len_building:
                    if  i + j + k + l == length_bulid:
                        return i, j, k, l



def suitable_building_length(len_, len_list=None):
    if len_list is None:
        len_list = length_combination()
    if len_ < MIN_LENGTH:
        return MIN_LENGTH
    if len_ > MAX_LENGTH:
        return MAX_LENGTH
    for x in len_list:
        if len_ >= x:
            res = x
        # else:
    return res


def detect_north(xx, yy):
    """检测北侧边"""
    slope_list = []
    index = 0
    while xx[index + 1] > xx[index]:
        slope_list.append(abs((yy[index + 1] - yy[index]) / (xx[index + 1] - xx[index])))
        index += 1
    return slope_list


def min_building_length(slope_list):
    """根据斜率返回满足条件的最短楼长"""
    n = len(slope_list)
    min_list = []
    for i in range(n):
        temp = max(math.ceil(MIN_DISTANCE / slope_list[i]), MIN_LENGTH + 15)
        if temp > MIN_LENGTH + 15:
            for x in length_combination():
                if temp <= x:
                    min_list.append(x)
                    break
        else:
            min_list.append(MIN_LENGTH + 15)
    return min_list


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
        # if is_in_ploygon(xx, yy, mid_point[i][0], mid_point[i][1]):
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


def is_in_polygon(xx, yy, x, y):
    """
    # Judge whether the point is in the ploygon
    :param xx: x axis
    :param yy: y axis
    :param x: x axis of point to be judged
    :param y: y axis of point to be judged
    :return: True: If point to be judged is in the polygon
             False: If point to be judged is out of the polygon
    """

    flag = -1
    try:
        n = xx.shape[0]
    except:
        n = len(xx)

    for i in range(1, n):
        a1, b1 = 0, 1 / y
        a2, b2 = coe_of_line(xx[i - 1], yy[i - 1], xx[i], yy[i])
        x_i, y_i = coord_of_intersection(a1, b1, a2, b2)
        if x_i >= x:
            if min(yy[i], yy[i - 1]) < y_i < max(yy[i], yy[i - 1]):
                flag *= -1
            elif max(abs(yy[i] - y_i), abs(yy[i - 1] - y_i)) < 1e-3:
                continue
            elif abs(y_i - max(yy[i], yy[i - 1])) < 1e-3:
                flag = -1 * flag
            elif abs(y_i - min(yy[i], yy[i - 1])) < 1e-3:
                continue
    if flag == 1:
        return True
    return False


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
        if d_A * d_B == 0:
            continue
        sin_theta = Vec_Cross / (d_A * d_B)
        if sin_theta == 0:
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


def x_coord_of_west_line(xx, yy):
    a2, b2 = coe_of_line(xx[0], yy[0], xx[1], yy[1])
    a3, b3 = coe_of_line(xx[0], yy[0], xx[-2], yy[-2])
    for x_ in np.linspace(xx[0], xx[0] + MIN_DISTANCE, 100):
        x_up, y_up = coord_of_intersection(1 / x_, 0, a2, b2)
        x_down, y_down = coord_of_intersection(1 / x_, 0, a3, b3)
        if abs(y_up - y_down) >= MIN_DISTANCE:
            # 如果满足楼宽，则返回起始点
            return x_


def plot_vertical_line(xx, yy, min_list):
    """
    根据多边形形状画竖线
    :param xx:
    :param yy:
    :return:
    """
    slope_list = detect_north(xx, yy)
    x_left, x_right = x_coord_of_west_line(xx, yy), xx[1]
    x_ = x_left
    x_list = [x_]
    while x_ <= x_right:
        plt.plot([x_, x_], [min(yy), max(yy)], color='grey', linestyle='--', linewidth=0.5)
        x_ += min(min_list[0] + 15, MAX_LENGTH)
        x_list.append(x_)

    for i in range(1, len(slope_list)):
        x_left, x_right = x_list[-1], xx[i + 1]
        x_ = x_left
        while x_ <= x_right:
            plt.plot([x_, x_], [min(yy), max(yy)], color='grey', linestyle='--', linewidth=0.5)
            x_ += min(min_list[i] + 15, MAX_LENGTH)
            x_list.append(x_)

    return x_list


def coord_of_intersection(a1, b1, a2, b2):
    """ Calculate coordinates of intersection of line a1*x + b1*y = 1 and a2*x + b2*y = 1 """
    A = np.array([[a1, b1], [a2, b2]])
    b = np.array([1, 1])
    x = np.linalg.solve(A, b)
    return x[0], x[1]


def coe_of_line(x1, y1, x2, y2):
    """ Calculate the coefficients of straight lines passing through points (x1, y1) and (x2, y2) """
    A = np.array([[x1, y1], [x2, y2]])
    b = np.array([1, 1])
    if x1 == x2:
        return 1 / x1, 0
    x = np.linalg.solve(A, b)
    return x[0], x[1]


def plot_building_at_north(xx, yy, x_list):
    """从北至南排楼"""
    slope_list = detect_north(xx, yy)
    min_list = min_building_length(slope_list)
    index = 0
    x_i = x_list[index]  # 设定起始点
    position_list = []  # list用于储存排楼走向
    full_coord_list = []
    for i in range(len(min_list)):
        while x_i <= xx[i + 1]:
            coord_list = []
            x_i = x_list[index]
            a2, b2 = coe_of_line(xx[i], yy[i], xx[i + 1], yy[i + 1])
            _, y_i = coord_of_intersection(1 / x_i, 0, a2, b2)  # 计算竖线与北侧边的交点坐标
            if is_in_polygon(xx, yy, x_i + min_list[i], y_i):
                origin_length = x_list[index + 1] - x_list[index]
                # x_left, x_right = x_i, x_i + min_list[i]
                x_left, x_right = x_i, x_list[index + 1]
                y_up, y_down = y_i, y_i - MIN_DISTANCE
                building_length = suitable_building_length(origin_length - 15)
                x_right_first = x_i + building_length
                # plt.plot([x_left, x_right, x_right, x_left, x_left],
                #          [y_up, y_up, y_down, y_down, y_up],
                #          color='k', linewidth=0.5)  # 画西北第一排楼
                position_list.append('right')
                coord_list.append([x_left, x_right_first, y_up, y_down])

            elif is_in_polygon(xx, yy, x_i - min_list[i], y_i):
                if position_list[-1] == 'right':
                    position_list.append('blank')
                    dividing_index = index
                    continue
                origin_length = x_i - x_list[index - 1]
                x_left, x_right = x_list[index - 1], x_i
                y_up, y_down = y_i, y_i - MIN_DISTANCE
                if position_list[-1] == 'blank':
                    x_left_first = x_left
                else:
                    building_length = suitable_building_length(origin_length - 15)
                    x_left_first = x_right - building_length
                # plt.plot([x_right, x_right, x_left, x_left, x_right],
                #           [y_up, y_down, y_down, y_up, y_up],
                #           color='k', linewidth=0.5)
                position_list.append('left')
                coord_list.append([x_left_first, x_right, y_up, y_down])

            x_left_, x_right_ = x_left, x_right  # 记录楼左侧横坐标与右侧横坐标以供之后使用

            y_up = y_down - BUILDING_INTERVAL
            y_down = y_up - MIN_DISTANCE
            y_down_lowest = y_down
            while is_in_polygon(xx, yy, x_left, y_down) or is_in_polygon(xx, yy, x_right, y_down):
                temp = []
                for j in range(1, len(xx)):
                    a2, b2 = coe_of_line(xx[j - 1], yy[j - 1], xx[j], yy[j])
                    x_i, y_i = coord_of_intersection(0, 1 / y_down, a2, b2)
                    if min(xx[j - 1], xx[j]) <= x_i <= max(xx[j - 1], xx[j]):
                        temp.append(x_i)
                x_i_left, x_i_right = min(temp), max(temp)  # y = y_down水平线与多边形的交点的横坐标
                if x_left <= x_i_left <= x_right <= x_i_right:
                    building_length = x_right - x_i_left
                    if building_length < MIN_LENGTH:
                        break
                    building_length = suitable_building_length(building_length, length_combination())
                    x_left = x_i_left
                    x_right = x_left + building_length
                elif x_i_left <= x_left <= x_i_right <= x_right:
                    building_length = x_i_right - x_left
                    if building_length < MIN_LENGTH:
                        break
                    building_length = suitable_building_length(building_length, length_combination())
                    x_right = x_i_right
                    x_left = x_right - building_length
                elif x_left <= x_i_left <= x_i_right <= x_right:
                    building_length = x_i_right - x_i_left
                    if building_length < MIN_LENGTH:
                        break
                    building_length = suitable_building_length(building_length, length_combination())
                    x_left = x_i_left
                    x_right = x_left + building_length
                else:
                    building_length = min_list[i]

                building_length = min(building_length, min_list[i])
                if building_length < MIN_LENGTH:
                    y_up = y_down - BUILDING_INTERVAL
                    y_down = y_up - MIN_DISTANCE
                    continue
                # print(building_length)
                # plt.plot([x_left, x_right, x_right, x_left, x_left],
                #          [y_up, y_up, y_down, y_down, y_up],
                #          color='k', linewidth=0.5)
                coord_list.append([x_left, x_right, y_up, y_down])
                y_down_lowest = y_down
                y_up = y_down - BUILDING_INTERVAL
                y_down = y_up - MIN_DISTANCE

            index += 1
            x_i = x_list[index]
            x_left, x_right = x_left_, x_right_

            for k in range(1, len(xx)):
                a2, b2 = coe_of_line(xx[k - 1], yy[k - 1], xx[k], yy[k])
                _, y_1 = coord_of_intersection(1 / x_left, 0, a2, b2)
                _, y_2 = coord_of_intersection(1 / x_right, 0, a2, b2)
                if min(yy[k - 1], yy[k]) <= y_1 <= max(yy[k - 1], yy[k]):
                    y_left = y_1
                if min(yy[k - 1], yy[k]) <= y_2 <= max(yy[k - 1], yy[k]):
                    y_right = y_2

            if y_left + MIN_DISTANCE > y_down_lowest - 15:
                pass
            elif y_right + MIN_DISTANCE > y_down_lowest - 15:
                pass
            else:
                y_down = max(y_left, y_right)
                y_up = y_down + MIN_DISTANCE
                # plt.plot([x_left, x_right, x_right, x_left, x_left],
                #      [y_up, y_up, y_down, y_down, y_up],
                #      color='k', linewidth=0.5)
                coord_list.append([x_left, x_right, y_up, y_down])

            if len(coord_list) >= 3:
                coord_list[-2][2] = 1 / 2 * (coord_list[-1][2] + coord_list[-3][2])
                coord_list[-2][3] = coord_list[-2][2] - MIN_DISTANCE
            # for coord in coord_list:
            #     x_left, x_right, y_up, y_down = coord
            # plt.plot([x_left, x_right, x_right, x_left, x_left],
            #          [y_up, y_up, y_down, y_down, y_up],
            #          color='k', linewidth=0.5)
            if coord_list:
                full_coord_list.append(coord_list)

    return full_coord_list


def rearrange_building(full_coord_list):
    """自南向北从第二栋楼开始，重新按楼间距排楼"""
    n = len(full_coord_list)
    for i in range(1, n):
        for j in range(1, len(full_coord_list[i])):
            min_distance = 2 ** 31
            for k in range(len(full_coord_list[i - 1])):
                distance = abs(full_coord_list[i][j][2] - full_coord_list[i - 1][k][2])
                if distance < min_distance:
                    min_distance = distance
                    min_index = k
            if min_distance < 42:
                full_coord_list[i][j][2] = full_coord_list[i - 1][min_index][2] + 42
                full_coord_list[i][j][3] = full_coord_list[i][j][2] - MIN_DISTANCE

    # for coord_list in full_coord_list:
    #     for coord in coord_list:
    #         x_left, x_right, y_up, y_down = coord
    #         plt.plot([x_left, x_right, x_right, x_left, x_left],
    #                  [y_up, y_up, y_down, y_down, y_up],
    #                  color='k', linewidth=0.5)

    return full_coord_list


def rearrange_last_building(full_coord_list, xx, yy):
    """根据需求重排最南端的楼"""
    full_info_list = []  # 存储楼的坐标信息与层数
    for coord_list in full_coord_list:
        # 由于按照间距重排楼很可能造成最南端楼的长度小于理想楼长，故重新计算最南端楼长
        if len(coord_list) < 2:  # 考虑从北至南楼数大于2的情况
            continue
        bound_x_left = sorted(coord_list, key=lambda s: s[0])[0][0]  # 计算x_left的边界值
        bound_x_right = sorted(coord_list, key=lambda s: -s[1])[0][1]  # 计算x_right的边界值
        x_left, x_right, y_up, y_down = coord_list[-1]
        x_list_up = []
        x_list_down = []
        for i in range(1, len(xx)):
            # 求解与y = y_up和y = y_down相交的边的横坐标情况
            a2, b2 = coe_of_line(xx[i - 1], yy[i - 1], xx[i], yy[i])
            x_1, y_up = coord_of_intersection(0, 1 / y_up, a2, b2)
            x_2, y_down = coord_of_intersection(0, 1 / y_down, a2, b2)
            if min(xx[i - 1], xx[i]) <= x_1 <= max(xx[i - 1], xx[i]):
                x_list_up.append(x_1)
            if min(xx[i - 1], xx[i]) <= x_2 <= max(xx[i - 1], xx[i]):
                x_list_down.append(x_2)
        # 初始化
        x_left_up, x_left_down, x_right_up, x_right_down = -100, -100, -100, -100  # 初始化
        # 如果最后一栋楼长度还可以增加而不影响其他条件，则更新最后一栋楼的左右横坐标
        for x_1 in x_list_up:
            if x_1 <= bound_x_left <= x_left:
                x_left_up = bound_x_left
            if bound_x_left < x_1 <= x_left:
                x_left_up = x_1
            if x_right < x_1 <= bound_x_right:
                x_right_up = x_1
            if x_right <= bound_x_right <= x_1:
                x_right_up = bound_x_right
        for x_2 in x_list_down:
            if x_2 <= bound_x_left <= x_left:
                x_left_down = bound_x_left
            if bound_x_left < x_2 <= x_left:
                x_left_down = x_2
            if x_right < x_2 <= bound_x_right:
                x_right_down = x_2
            if x_right <= bound_x_right <= x_2:
                x_right_down = bound_x_right
        # 尽可能在多边形内画出最大楼
        if is_in_polygon(xx, yy, x_left_up, y_up) and is_in_polygon(xx, yy, x_left_up, y_down):
            x_left = x_left_up
        if is_in_polygon(xx, yy, x_left_down, y_up) and is_in_polygon(xx, yy, x_left_down, y_down):
            x_left = x_left_down
        if is_in_polygon(xx, yy, x_right_up, y_up) and is_in_polygon(xx, yy, x_right_up, y_down):
            x_right = x_right_up
        if is_in_polygon(xx, yy, x_right_down, y_up) and is_in_polygon(xx, yy, x_right_down, y_down):
            x_right = x_right_down
        # 更新坐标信息
        coord_list[-1] = [x_left, x_right, y_up, y_down]

    return full_coord_list


def rearrange_to_fit_plotRatio(full_coord_list, xx, yy):
    full_info_list = full_info_of_building(full_coord_list)
    print(full_info_list)
    sorted_list = []  # 将info列表重新按照楼的纵坐标排列
    index_1 = 0
    # 将info列表拆分
    for info_list in full_info_list:
        index_2 = 0
        for info in info_list:
            sorted_list.append(info + [index_1, index_2])
            index_2 += 1
        index_1 += 1
    # 将列表按纵坐标升序排列
    sorted_list = sorted(sorted_list, key=lambda s: s[3])
    iter = 0
    count = 0

    def _plot_ratio(sorted_list):
        res = 0
        for info in sorted_list:
            res += info[4] * info[5] * 0.8 / AREA
        return res

    while _plot_ratio(sorted_list) > plot_Ratio:
        count += 1
        if count > 300:
            break
        # 当容积率大于阈值时，执行循环
        try:
            info_this_building = sorted_list[iter]  # 选择对应的楼
        except:
            break
        sorted_list.remove(info_this_building)  # 将其移除列表方便对其进行修改
        x_left_t, x_right_t, y_up_t, y_down_t, _, _, index_1_t, index_2_t = info_this_building
        if info_this_building[-1] == 0:
            # 如果需要判断最北侧的楼房，则跳出循环
            sorted_list.append(info_this_building)
            break
        for info in sorted_list:
            # 找出这栋楼房的北侧距离最近的第一座楼房
            if info[-1] == info_this_building[-1] - 1 and info[-2] == info_this_building[-2]:
                info_last_building = info
                x_left_l, x_right_l, y_up_l, y_down_l, _, _, _, _ = info_last_building
                break
        building_height = y_down_l - y_up_t  # 返回这栋楼房的高度
        floor = math.floor(building_height / UNIT_HIGH)  # 返回这栋楼房的楼层数
        if floor < 3:
            # 如果楼层数小于3，则将这栋楼房删除
            continue
        else:
            building_length = suitable_building_length(x_right_t - x_left_t)
            print(building_length)
            if building_length >= 28:
                # 如果楼房的长度超过28米，则东西各减5米
                info_this_building[0] += 5
                info_this_building[1] -= 5
                info_this_building[4] -= 10 * MIN_DISTANCE
            if floor <= 18:
                # 如果楼房层数不足6层，则不压缩这栋楼的楼高
                info_this_building[5] = floor
                move_distance = 0
                # iter += 1
                # sorted_list.append(info_this_building)
                # sorted_list = sorted(sorted_list, key=lambda s: s[3])
                # continue
            else:
                # 如果楼房层数大于6层，则将这栋楼压缩至6层
                info_this_building[5] = 18
                move_distance = y_down_l - y_up_t - 18 * UNIT_HIGH  # 计算需要北移的距离
                info_this_building[2] += move_distance
                info_this_building[3] += move_distance
            building_group = []  # building_group用于记录同一列的在南侧的楼房组
            for info in sorted_list:
                if info[-1] > index_2_t and info[-2] == index_1_t:
                    # 如果楼房是在同一列的南侧，则向北移动
                    info[2] += move_distance
                    info[3] += move_distance
                    building_group.append(info)
            if not building_group:
                # 如果该楼房已是同一列的最南一栋，则直接进行下一个循环
                sorted_list.append(info_this_building)
                sorted_list = sorted(sorted_list, key=lambda s: s[3])
                iter += 1
                continue
            # 将同一列的楼房自北向南排列
            building_group = sorted(building_group, key=lambda s: s[-1])
            x_left_bound, x_right_bound = 2 ** 31, -2 ** 31
            for info in building_group:
                # 找到横坐标的阈值
                x_left_bound = min(x_left_bound, info[0])
                x_right_bound = max(x_right_bound, info[1])
            # x_left_bound, x_right_bound = building_group[-1][0], building_group[-1][1]
            # 判断南向余地是否可以排楼
            temp_y = building_group[-1][3] - height_limit - MIN_DISTANCE
            is_new_building = False
            if is_in_polygon(xx, yy, x_left_bound, temp_y) and is_in_polygon(xx, yy, x_right_bound, temp_y):
                # 先判断是否可在边界以上排一栋高楼
                y_down_new = temp_y
                y_up_new = y_down_new + MIN_DISTANCE
                rectangle_area = (x_right_bound - x_left_bound) * MIN_DISTANCE
                new_building_info = [x_left_bound, x_right_bound, y_up_new, y_down_new, rectangle_area, 24,
                                     building_group[-1][-2], building_group[-1][-1] + 1]
                building_group.append(new_building_info)
                sorted_list.append(new_building_info)
                is_new_building = True
                # 下一个循环从新加进的楼开始执行
                # 如果不能在边界以上排楼，则判断是否可以在边界上排楼
            for i in range(1, len(xx)):
                a2, b2 = coe_of_line(xx[i - 1], yy[i - 1], xx[i], yy[i])
                _, y_left_i = coord_of_intersection(1 / x_left_bound, 0, a2, b2)
                _, y_right_i = coord_of_intersection(1 / x_right_bound, 0, a2, b2)
                if min(yy[i - 1], yy[i]) <= y_left_i <= max(yy[i - 1], yy[i]):
                    y_left_bound = y_left_i
                if min(yy[i - 1], yy[i]) <= y_right_i <= max(yy[i - 1], yy[i]):
                    y_right_bound = y_right_i

            y_down_new = max(y_left_bound, y_right_bound)
            y_up_new = y_down_new + MIN_DISTANCE

            building_height = building_group[-1][3] - y_up_new
            if building_height < 6*UNIT_HIGH:
                sorted_list.append(info_this_building)
                sorted_list = sorted(sorted_list, key=lambda s: s[3])
                if is_new_building:
                    iter = sorted_list.index(new_building_info)
                else:
                    iter += 1
                continue
            # if building_height < 6 * 3.3:
            #     # 如果排的楼层数不足6楼，则不排楼
            #     sorted_list.append(info_this_building)
            #     sorted_list = sorted(sorted_list, key=lambda s: s[3])
            #     iter += 1
            #     continue
            # else:
            floor = math.floor(building_height/UNIT_HIGH)
            rectangle_area = (x_right_bound - x_left_bound) * MIN_DISTANCE
            new_building_info = [x_left_bound, x_right_bound, y_up_new, y_down_new, rectangle_area, floor,
                                 building_group[-1][-2], building_group[-1][-1] + 1]
            # print(new_building_info)
            sorted_list.append(info_this_building)
            sorted_list.append(new_building_info)
            sorted_list = sorted(sorted_list, key=lambda s: s[3])
            iter = sorted_list.index(new_building_info)  # 下一个循环从新加进的楼开始进行判断
        iter += 1

    sorted_list = sorted(sorted_list, key=lambda s: (s[-2], s[-1]))
    # 以下部分为根据sorted_list还原full_info_list和full_coord_list
    full_info_list = []
    full_coord_list = []
    temp_info_list = []
    temp_coord_list = []
    temp_info_list.append(sorted_list[0][0:-2])
    temp_coord_list.append(sorted_list[0][0:4])
    for i in range(1, len(sorted_list)):
        if sorted_list[i][-2] == sorted_list[i - 1][-2] and i < len(sorted_list) - 1:
            temp_info_list.append(sorted_list[i][0:-2])
            temp_coord_list.append(sorted_list[i][0:4])
        elif sorted_list[i][-2] == sorted_list[i - 1][-2] and i == len(sorted_list) - 1:
            temp_info_list.append(sorted_list[i][0:-2])
            temp_coord_list.append(sorted_list[i][0:4])
            full_info_list.append(temp_info_list)
            full_coord_list.append(temp_coord_list)
        elif sorted_list[i][-2] != sorted_list[i - 1][-2] and i < len(sorted_list) - 1:
            full_info_list.append(temp_info_list)
            full_coord_list.append(temp_coord_list)
            temp_info_list = []
            temp_coord_list = []
            temp_info_list.append(sorted_list[i][0:-2])
            temp_coord_list.append(sorted_list[i][0:4])
        else:
            full_info_list.append(temp_info_list)
            full_coord_list.append(temp_coord_list)
            temp_info_list = []
            temp_coord_list = []
            temp_info_list.append(sorted_list[i][0: -2])
            temp_coord_list.append(sorted_list[i][0:4])
            full_info_list.append(temp_info_list)
            full_coord_list.append(temp_coord_list)
    full_coord_list = [[[int(x) for x in full_coord_list[i][j]]
                        for j in range(len(full_coord_list[i]))] for i in range(len(full_coord_list))]
    full_info_list = [[[int(x) for x in full_info_list[i][j]]
                       for j in range(len(full_info_list[i]))] for i in range(len(full_info_list))]
    full_coord_list = remove_same(full_coord_list)
    full_info_list = remove_same(full_info_list)
    return full_info_list, full_coord_list


def add_num_build(full_info_list):
    full_all_list = []
    num = 0
    for coord_list in full_info_list:
        for coord in coord_list:
            num += 1
            coord.append(num)
        # coord_list.append(coord)
        full_all_list.append(coord_list)
    return full_all_list


def get_ret_posi(full_coord_list):
    x_fir = []
    len_bui = []
    y_fir = []
    for coord_list in full_coord_list:
        for coord in coord_list:
            x_left, x_right, y_up, y_down = coord
            x_fir.append(x_left)
            y_fir.append(y_up)
            len_bui.append(x_right - x_left)
            # plt.plot([x_left, x_right, x_right, x_left, x_left],
            #          [y_up, y_up, y_down, y_down, y_up],
            #          color='k', linewidth=0.5)

    return x_fir, y_fir, len_bui

def replot_build(AREA, full_all_list, full_info_list):
    while get_build_ratio(AREA, full_info_list) > 2:
        for list1 in full_all_list:
            if len(list1) > 2:
                del list1[1]

    for list1 in full_all_list:
        n=np.shape(list1)[0]
        len_fir_end=list1[0][2]-list1[-1][2]
        y_change= len_fir_end / (n-1)
        for ii in range(n):
            if ii != 0 or ii != n-1:
                list1[ii][2]=list1[0][2]-ii*y_change
                list1[ii][3]=list1[0][3]-ii*y_change

    y_fir = []
    x_fir = []
    len_bui = []
    for list1 in full_all_list:
        for list2 in list1:
            x=[list2[0], list2[1], list2[1], list2[0], list2[0]]
            y=[list2[2], list2[2], list2[3], list2[3], list2[2]]
            y_fir.append(list2[2])
            x_fir.append(list2[0])
            len_bui.append(list2[1] - list2[0])
            plt.plot(x,y)

    return full_all_list, get_build_ratio(AREA, full_info_list), x_fir, y_fir,len_bui



def polyg(xx, yy):
    """
    对给定坐标形成区域
    """
    aa = []
    for ii in range(len(xx) - 1):
        aa.append(xx[ii])
        aa.append(yy[ii])
        f1 = ""
        for ii in range(len(aa)):
            if ii % 2 == 0:

                f1 += str(aa[ii]) + " "
            else:
                f1 += str(aa[ii]) + "\n"
    return f1


def slope_cal(xx, yy):
    """求边的斜率"""
    slope_list = []
    for ii in range(len(xx) - 1):
        temp = (yy[ii + 1] - yy[ii]) / (xx[ii + 1] - yy[ii])
        slope_list.append(temp)
    return slope_list


def cal_len_edge(x1, y1, x2, y2):
    """求两点的长度"""
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def points(xx, yy, ff):
    """
    判断 ff 是否在 xx, yy 形成的区域内
    """
    pg = Polygon(np.genfromtxt(StringIO(ff)))  # 生成Polygon实例
    return geometry.Point([xx, yy]).within(pg)


def plot_tree_edge(xx, yy):
    """
    : xx: 边的横坐标
    : yy: 边的纵坐标
    : tree_distance: 树之间的距离
    """
    x_tree_edge = []
    y_tree_edge = []
    r_tree_edge = []
    for ii in range(len(xx) - 1):
        ll = cal_len_edge(xx[ii], yy[ii], xx[ii + 1], yy[ii + 1])
        num_tree = np.floor(ll / tree_distance)
        mod_dis = np.mod(ll, tree_distance)
        if mod_dis <= tree_distance / 2:
            tree_dis = ll / num_tree
            ll1 = 0
            x = xx[ii]
            y = yy[ii]
            while ll1 <= ll - mod_dis:
                x_tree_edge.append(x)
                y_tree_edge.append(y)
                ll1 += tree_dis
                r = random.uniform(1.5, 3)
                r_tree_edge.append(r)
                theta = np.arange(0, 2 * np.pi, 0.01)
                x1 = x + r * np.cos(theta)
                y1 = y + r * np.sin(theta)
                plt.plot(x1, y1, color='g')

                x += 15 * (xx[ii + 1] - xx[ii]) / ll
                y += 15 * (yy[ii + 1] - yy[ii]) / ll

        else:
            ll1 = 0
            x = xx[ii]
            y = yy[ii]
            while ll1 <= ll:
                ll1 += 15
                r = random.uniform(1.5, 3)
                x_tree_edge.append(x)
                y_tree_edge.append(y)
                r_tree_edge.append(r)
                theta = np.arange(0, 2 * np.pi, 0.01)
                x1 = x + r * np.cos(theta)
                y1 = y + r * np.sin(theta)
                plt.plot(x1, y1, color='g')

                x += 15 * (xx[ii + 1] - xx[ii]) / ll
                y += 15 * (yy[ii + 1] - yy[ii]) / ll

    return x_tree_edge, y_tree_edge, r_tree_edge


def plot_tree_region(xx, yy):
    """
    画区域内的树
    : xx: 退线之后点的横坐标
    : yy: 退线之后点的纵坐标
    : num_tree: 要种树的量
    """
    x_max = max(xx)
    x_min = min(xx)
    y_max = max(yy)
    y_min = min(yy)

    x_tree_region = []
    y_tree_region = []
    r_tree_region = []
    while len(y_tree_region) <= num_tree:
        x2 = random.uniform(x_min, x_max)
        y2 = random.uniform(y_min, y_max)
        if points(x2, y2, polyg(xx, yy)) == True:

            temp = []
            for ll in range(len(len_bui)):
                xx_re = np.array(
                    [x_fir[ll] - 15, x_fir[ll] + len_bui[ll] + 15, x_fir[ll] + len_bui[ll] + 15, x_fir[ll] - 15,
                     x_fir[ll] - 15])
                yy_re = [y_fir[ll] + 15, y_fir[ll] + 15, y_fir[ll] - 12 - 15, y_fir[ll] - 12 - 15, y_fir[ll] - 12 - 15]

                temp.append(points(x2, y2, polyg(xx_re, yy_re)))
            if sum(temp) == 0:
                y_tree_region.append(y2)
                x_tree_region.append(x2)
                r = random.uniform(1.5, 3)
                r_tree_region.append(r)
                theta = np.arange(0, 2 * np.pi, 0.01)
                x1 = x2 + r * np.cos(theta)
                y1 = y2 + r * np.sin(theta)
                plt.plot(x1, y1, color='g')

    return x_tree_region, y_tree_region, r_tree_region

def get_htype(floor):
    if floor == 6:
        return 1
    else:
        return 2

def output_infor(full_all_list,lon,lat):
    
    points = []
    for idx,x in enumerate(lon):
        points.append(dict(rx=x,ry=lat[idx]))
    
    landBoundary = dict(points=points)

    trees_edge = []
    for idx,x in enumerate(x_tree_edge):
        trees_edge.append(dict(tx=XY_to_miller(x, y_tree_edge[idx])[0],
        ty=XY_to_miller(x, y_tree_edge[idx])[1],tr=r_tree_edge[idx]))

    trees_region = []
    for idx,x in enumerate(x_tree_region):
        trees_region.append(dict(tx=XY_to_miller(x,y_tree_region[idx])[0],
        ty=XY_to_miller(x,y_tree_region[idx])[1],tr=r_tree_region[idx]))
    appendageArrange = dict(trees_edge=trees_edge,trees_region=trees_region)
    
    houseArrangeList = []
    
    for ha_row in full_all_list:
        for ha in ha_row:
            temp=get_combination(np.ceil(ha[1]-ha[0]))
            detail=[]
            
            for l in temp:
                detail.append(dict(dx=l,dy=12))

            houseArrangeList.append(dict(x=XY_to_miller((ha[0]+ha[1])/2,(ha[2]+ha[3])/2)[0],
            y=XY_to_miller((ha[0]+ha[1])/2,(ha[2]+ha[3])/2)[1],z=ha[5]*3.3,
                                        w=np.ceil(ha[1]-ha[0]),h=12,h_type=get_htype(ha[5]),floors=ha[5],
                                        buildingNumber=ha[6],detail=detail))
            

    landArrange = dict(houseArrange=houseArrangeList,
                        landBoundary=landBoundary,
                        appendageArrange=appendageArrange)
                    

    return dict(landArrange=landArrange)



if __name__ == '__main__':
    # xx = 0.5 * np.array([100, 350, 650, 800, 600, 510, 150, 100])
    # yy = 0.5 * np.array([700, 770, 690, 610, 40, 10, 80, 700])
    xx = np.array([0, 150, 250, 350, 200, 110, 20, 0])
    yy = np.array([340, 440, 360, 300, 70, 60, 80, 340])
    NORTH_WEST = np.array([[100], [770]])
    NORTH_EAST = np.array([[350], [690]])

    plt.plot(xx, yy)
    xx, yy = delVertex(xx, yy, threshold=0.005)
    plt.plot(xx, yy)
    x_scale, y_scale = scale(xx, yy, sec_dis=10)
    plt.plot(x_scale, y_scale)
    slope_list = detect_north(x_scale, y_scale)
    north_building_len_list = min_building_length(slope_list)
    # print(north_building_len_list)
    x_list = plot_vertical_line(x_scale, y_scale, north_building_len_list)
    full_coord_list = plot_building_at_north(x_scale, y_scale, x_list)
    full_coord_list = rearrange_building(full_coord_list)
    full_coord_list = rearrange_last_building(full_coord_list, x_scale, y_scale)
    full_info_list, full_coord_list = rearrange_to_fit_plotRatio(full_coord_list, x_scale, y_scale)
    plot_building(full_coord_list)
    # print('total plot ratio =', total_plot_ratio(full_info_list))
    # print(full_info_list)
    full_all_list = add_num_build(full_info_list)
    x_fir, y_fir, len_bui = get_ret_posi(full_coord_list)
    full_all_list, build_ratio, x_fir, y_fir, len_bui = replot_build(AREA, full_all_list, full_info_list)
    x_tree_edge, y_tree_edge, r_tree_edge = plot_tree_edge(xx, yy)
    x_tree_region, y_tree_region, r_tree_region = plot_tree_region(x_scale, y_scale)
    # print(length_combination())
    print(build_ratio)
    # print('houseArrange:')
    # print(full_all_list)

    lon,lat = get_lonlat_coor(xx,yy)
    output= output_infor(full_all_list, lon, lat)

    plt.show()

# %%
