# %%
import numpy as np
import matplotlib.pyplot as plt

xx=np.array([0,150,250,350, 200,110, 10,0])
yy=np.array([340,400,360,-20,-70 ,-120, 80,340])
sd = -10 # safe distance


# def isInPloygon(xx, yy, x, y):
#     """
#     # Judge whether the point is in the ploygon
#     :param xx: x axis
#     :param yy: y axis
#     :param x: x axis of point to be judged
#     :param y: y axis of point to be judged
#     :return: True: If point to be judged is in the polygon
#              False: If point to be judged is out of the polygon
#     """
#
#     flag = -1
#     for i in range(1, xx.shape[0]):
#         if x < max(xx[i], xx[i - 1]) and (min(yy[i], yy[i - 1]) < y < max(yy[i], yy[i - 1])):
#             flag = -1 * flag
#         elif yy[i] == y and yy[i - 1] == y and x < max(xx[i], xx[i - 1]):
#             continue
#         elif y == max(yy[i], yy[i - 1]) and x < max(xx[i], xx[i - 1]):
#             flag = -1 * flag
#         elif y == min(yy[i], yy[i - 1]) and x < max(xx[i], xx[i - 1]):
#             continue
#     if flag == 1:
#         return True
#     return False

# Calculate slope
# def slope_cal(x_cor,y_cor):
#     xx=x_cor
#     yy=y_cor
#     xx_dif=np.diff(xx)
#     yy_dif=np.diff(yy)
#     slope=yy_dif/xx_dif
#     slope=np.append(slope,slope[0])
#     return slope
#
# slope=slope_cal(xx,yy)

# Calculate slope
def slope_cal(x_cor, y_cor):
    xx = x_cor
    yy = y_cor
    xx_dif = np.diff(xx)
    yy_dif = np.diff(yy)
    slope = yy_dif/xx_dif
    slope = np.append(slope, slope[0])
    return slope

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

def coord_of_intersection(a1, b1, a2, b2):
    """ Calculate coordinates of intersection of line a1*x + b1*y = 1 and a2*x + b2*y = 1 """
    A = np.array([[a1, b1], [a2, b2]])
    b = np.array([[1], [1]])
    x = np.linalg.solve(A, b)
    return x[0], x[1]


def coe_of_line(x1, y1, x2, y2):
    """ Calculate the coefficients of straight lines passing through points (x1, y1) and (x2, y2) """
    A=np.array([[x1, y1], [x2, y2]])
    b = np.array([[1], [1]])
    x = np.linalg.solve(A, b)
    return x[0], x[1]


def coord_of_horizon(xx, yy, h):
    """
    :param xx: x - axis of given polygon
    :param yy: y - axis of given polygon
    :param h: y - axis of the horizon line
    :return: set of the coords of the intersection of horizon line and polygon
             (may contain duplicate data)
    """
    a2 = 0
    b2 = 1 / h
    res = []
    for i in range(len(xx) - 1):
        x1, y1, x2, y2 = xx[i], yy[i], xx[i + 1], yy[i + 1]
        a1, b1 = coe_of_line(x1, y1, x2, y2)
        x_inter, _ = coord_of_intersection(a1, b1, a2, b2)
        if x1 <= x_inter <= x2:
            res.append(x_inter)

    res = sorted(res)
    return res


def adjust_horizon_line(xx, yy, threshold):
    """
    Adjust the horizon line to satisfy the threshold condition
    :param xx: x - axis of the given polygon
    :param yy: y - axis of the given polygon
    :param threshold: minimum critical length of row
    :return: abscissa of intersection
    """
    y_max = np.max(yy)
    y_min = np.min(yy)
    for h in np.linspace(y_max, y_min, 100):
        coord = coord_of_horizon(xx, yy, h)
        if coord[-1] - coord[0] >= threshold:
            break
    start_x_left = coord[0]
    start_x_right = coord[-1]
    return start_x_left, start_x_right

plt.plot(xx, yy)
xx, yy = delVertex(xx, yy, threshold = 0.005)
plt.plot(xx, yy)
scale_x, scale_y = scale(xx, yy, sec_dis = 10)
plt.plot(scale_x, scale_y)

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

max_x = np.max(scale_x)
max_y = np.max(scale_y)
min_x = np.min(scale_x)
min_y = np.min(scale_y)
mid_x = (max_x + min_x)/2
mid_y = (max_y + min_y)/2

# # Judge whether the point is in the ploygon
# coor = np.array([xx,yy])
# co0r = coor.T[:len(xx) - 1,:]
# # fig = plt.figure()
# # axes = fig.add_subplot(1)
# # p1 = plt.Polygon(cor)
# # axes.add_patch(p1)
# # plt.show()
# # from shapely import geometry
# import shapely.geometry as geometry
#
# def if_inPoly (polygon, Points):
#     line = geometry.LineString(polygon)
#     point = geometry.Point(Points)
#     polygon = geometry.Polygon(line)
#     return polygon.contains(point)
#
# pt2 = (400, 400)
# print(if_inPoly(co0r, pt2))

# %%
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


# %%
# Calculation of east-west energy rankings (approximate)
interval = 80
depth = 12
h_raw_num = np.ceil((max_y - min_y) / (interval + depth))
print(h_raw_num)
plt.plot(scale_x, scale_y)
y_max = np.max(scale_y)
y_min = np.min(scale_y)

for i in range(int(h_raw_num)+1):
    if y_max - 92 - 12 < y_min:
        break

    # intersec_posi_x1_total=[]
    # intersec_posi_x2_total=[]

    if i == 0:

        slope = slope_cal(scale_x, scale_y)

        if len(np.where(scale_y==np.max(scale_y))[0]) ==2:
            # y_max = y_max
            # intersec_posi_x1, intersec_posi_x2 = calculation_SN(scale_x, scale_y, y_max)
            intersec_posi_x1=scale_x[np.where(scale_y==np.max(scale_y))[0][0]] 
            intersec_posi_x2=scale_x[np.where(scale_y==np.max(scale_y))[0][1]]
            ll1 = np.abs(intersec_posi_x2 - intersec_posi_x1)

            plt.plot([intersec_posi_x1, intersec_posi_x2], [y_max, y_max])

            temp = arr_ew(ll1, len_bulid_total)

            mod_aera = []
            for ii in len_bulid_total[:len(temp)]:
                if ii + 5 > ll1:
                    break
                if len(temp[str(ii)]) == 4:
                    mod_aera.append(temp[str(ii)][3])
                if len(temp[str(ii)]) == 6:
                    mod_aera.append(temp[str(ii)][5])

            min_mod = np.where(mod_aera == min(mod_aera))

            if len(temp[str(len_bulid_total[min_mod[0][0]])]) == 4:

                num_build = temp[str(len_bulid_total[min_mod[0][0]])][1]
                for jj in list(range(int(num_build))):
                    x_build_start = intersec_posi_x1 + jj * (5 + temp[str(len_bulid_total[min_mod[0][0]])][0])
                    x_build_second = x_build_start + temp[str(len_bulid_total[min_mod[0][0]])][0]
                    x_build = np.hstack((x_build_start, x_build_second))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, x_build_start)

                    y_build = np.array([y_max, y_max, y_max - 12, y_max - 12, y_max])

                    plt.plot(x_build, y_build)

            elif len(temp[str(len_bulid_total[min_mod[0][0]])]) == 6:
                num_build = temp[str(len_bulid_total[min_mod[0][0]])][1]
                print(num_build)
                for jj in range(int(num_build + 1)):
                    x_build_start = intersec_posi_x1 + jj * (5 + temp[str(len_bulid_total[min_mod[0][0]])][0])
                    if jj < int(num_build):
                        x_build_second = x_build_start + temp[str(len_bulid_total[min_mod[0][0]])][0]
                    elif jj == int(num_build):
                        x_build_second = x_build_start + temp[str(len_bulid_total[min_mod[0][0]])][3][0]
                    x_build = np.hstack((x_build_start, x_build_second))
                    x_build = np.hstack((x_build, np.flipud(x_build)))
                    x_build = np.append(x_build, x_build_start)

                    y_build = np.array([y_max, y_max, y_max - 12, y_max - 12, y_max])

                    plt.plot(x_build, y_build)

            print('排楼方法为：' + str(temp[str(len_bulid_total[min_mod[0][0]])]))

        # intersec_posi_x1,intersec_posi_x2=adjust_horizon_line(scale_x,scale_y,min(len_bulid_total))
        else:
            for ii in np.linspace(0.01, 40, 20000):
                y_max = y_max - ii
                intersec_posi_x1, intersec_posi_x2 = calculation_SN(scale_x, scale_y, y_max)
                ll1 = np.abs(intersec_posi_x2 - intersec_posi_x1)
                if ll1 >= np.min(len_bulid_total):
                    break

            # y_max = y_max - 6
            # intersec_posi_x1, intersec_posi_x2 = calculation_SN(scale_x, scale_y, y_max)
            # if intersec_posi_x1 > intersec_posi_x2:
            #     x_temp=intersec_posi_x1
            #     intersec_posi_x1=intersec_posi_x2
            #     intersec_posi_x2=x_temp

            # intersec_posi_x1_total.append(intersec_posi_x1)
            # intersec_posi_x2_total.append(intersec_posi_x2)

            # ll1 = np.abs(intersec_posi_x2 - intersec_posi_x1)

            plt.plot([intersec_posi_x1, intersec_posi_x2], [y_max, y_max])

            # temp = arr_ew(ll1, len_bulid_total)

            # mod_aera = []
            # for ii in len_bulid_total[:len(temp)]:
            #     if ii + 5 > ll1:
            #         break
            #     if len(temp[str(ii)]) == 4:
            #         mod_aera.append(temp[str(ii)][3])
            #     if len(temp[str(ii)]) == 6:
            #         mod_aera.append(temp[str(ii)][5])

            # min_mod = np.where(mod_aera == min(mod_aera))

            # num_build = 1
            # for jj in list(range(int(num_build))):
            x_build_start = intersec_posi_x1
            x_build_second = x_build_start + np.min(len_bulid_total)
            x_build = np.hstack((x_build_start, x_build_second))
            x_build = np.hstack((x_build, np.flipud(x_build)))
            x_build = np.append(x_build, x_build_start)

            y_build = np.array([y_max, y_max, y_max - 12, y_max - 12, y_max])

            plt.plot(x_build, y_build)

            print('排楼方法为：' + str(np.min(len_bulid_total)))

    else:
        y_max = y_max - 92

        intersec_posi_x1, intersec_posi_x2 = calculation_SN(scale_x, scale_y, y_max)
        # if intersec_posi_x1 > intersec_posi_x2:
        #     x_temp=intersec_posi_x1
        #     intersec_posi_x1=intersec_posi_x2
        #     intersec_posi_x2=x_temp

        # intersec_posi_x1_total.append(intersec_posi_x1)
        # intersec_posi_x2_total.append(intersec_posi_x2)

        ll1 = np.abs(intersec_posi_x2 - intersec_posi_x1)
        plt.plot([intersec_posi_x1, intersec_posi_x2], [y_max, y_max])

        temp = arr_ew(ll1, len_bulid_total)

        mod_aera = []
        for ii in len_bulid_total:
            if ii + 5 > ll1:
                break
            if len(temp[str(ii)]) == 4:
                mod_aera.append(temp[str(ii)][3])
            if len(temp[str(ii)]) == 6:
                mod_aera.append(temp[str(ii)][5])

        min_mod = np.where(mod_aera == min(mod_aera))

        if len(temp[str(len_bulid_total[min_mod[0][0]])]) == 4:

            num_build = temp[str(len_bulid_total[min_mod[0][0]])][1]
            for jj in list(range(int(num_build))):
                x_build_start = intersec_posi_x1 + jj * (5 + temp[str(len_bulid_total[min_mod[0][0]])][0])
                x_build_second = x_build_start + temp[str(len_bulid_total[min_mod[0][0]])][0]
                x_build = np.hstack((x_build_start, x_build_second))
                x_build = np.hstack((x_build, np.flipud(x_build)))
                x_build = np.append(x_build, x_build_start)

                y_build = np.array([y_max, y_max, y_max - 12, y_max - 12, y_max])

                plt.plot(x_build, y_build)

        elif len(temp[str(len_bulid_total[min_mod[0][0]])]) == 6:
            num_build = temp[str(len_bulid_total[min_mod[0][0]])][1]
            print(num_build)
            for jj in range(int(num_build + 1)):
                x_build_start = intersec_posi_x1 + jj * (5 + temp[str(len_bulid_total[min_mod[0][0]])][0])
                if jj < int(num_build):
                    x_build_second = x_build_start + temp[str(len_bulid_total[min_mod[0][0]])][0]
                elif jj == int(num_build):
                    x_build_second = x_build_start + temp[str(len_bulid_total[min_mod[0][0]])][3][0]
                x_build = np.hstack((x_build_start, x_build_second))
                x_build = np.hstack((x_build, np.flipud(x_build)))
                x_build = np.append(x_build, x_build_start)

                y_build = np.array([y_max, y_max, y_max - 12, y_max - 12, y_max])

                plt.plot(x_build, y_build)

        print('排楼方法为：' + str(temp[str(len_bulid_total[min_mod[0][0]])]))

plt.show()

    # print(num_build)
