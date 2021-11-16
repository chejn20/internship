import numpy as np
import matplotlib.pyplot as plt

xx = np.array([0, 40, 100, 140, 150, 10, 0])
yy = np.array([140, 60, 150, 130, 10, 10, 140])
sd = -10 # safe distance

def isInPloygon(xx, yy, x, y):
    """
    # Judge whether the point is in the ploygon
    :param xx: x axis
    :param yy: y axis
    :param x: x axis of point to be judged
    :param y: y axis of point to be judged
    :return: True: If point to be judged is in the polygon
             False: If point to be judged is out of the polygon
    """

plt.figure()
plt.plot(xx, yy)
xx_dif = np.diff(xx)
yy_dif = np.diff(yy)

def isInPloygon(xx, yy, x, y):
    # Judge whether the point is in the ploygon
    flag = -1
    for i in range(1, xx.shape[0]):
        if x < max(xx[i], xx[i - 1]) and (min(yy[i], yy[i - 1]) < y < max(yy[i], yy[i - 1])):
            flag = -1*flag
        elif yy[i] == y and yy[i - 1] == y and x < max(xx[i], xx[i - 1]):
            continue
        elif y == max(yy[i], yy[i - 1]) and x < max(xx[i], xx[i - 1]):
            flag = -1*flag
        elif y == min(yy[i], yy[i - 1]) and x < max(xx[i], xx[i - 1]):
            continue
    if flag == 1:
        return True
    return False

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


def delVertex(xx, yy, threshold=0.05):
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
        if isInPloygon(xx, yy, mid_point[i][0], mid_point[i][1]):
            if ratio[i] < threshold:
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
        date.append([x[i], y[i]])
    num = len(data)
    xx = []
    yy = []
    for i in range(num):
        x1 = data[(i) % num][0] - data[(i - 1) % num][0]
        y1 = data[(i) % num][0] - data[(i - 1) % num][1]
        x2 = data[(i) % num][0] - data[(i - 1) % num][0]
        y2 = data[(i) % num][0] - data[(i - 1) % num][1]

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

xx2, yy2 = delVertex(xx, yy)
plt.plot(xx2, yy2)

slope = yy_dif / xx_dif
slope = np.append(slope, slope[0])

# Delete the maximum angle
# deg1 = []
# for ii in list(range(len(slope) - 1)):
#     # print(ii)
#     ang = (slope[ii + 1] - slope[ii]) / (1 + slope[ii + 1] * slope[ii])
#     deg = deg = np.arctan(ang) * 180 / np.pi
#     if deg < 0:
#         deg = deg + 180
#     deg1.append(deg)
# pos = np.where(deg1 == np.max(deg1))
# xx[pos[0]+1]
# xx1 = np.delete(xx, pos[0] + 1)
# yy1 = np.delete(yy, pos[0] + 1)
# plt.plot(xx1, yy1)


# line drawback
# max_x = np.max(xx)
# max_y = np.max(yy)
# min_x = np.min(xx)
# min_y = np.min(yy)
# mid_x = (max_x + min_x) / 2
# mid_y = (max_y + min_y) / 2
# xx2 = []
# yy2 = []
# for ii in list(range(len(xx))):
#     #print(ii)
#     if xx[ii] <= mid_x:
#         xx2.append(xx[ii] + 10)
#     else:
#         xx2.append(xx[ii] - 10)
#     if yy[ii] <= mid_y:
#         yy2.append(yy[ii] + 10)
#     else:
#         yy2.append(yy[ii] - 10)
#
# plt.plot(xx2, yy2)

# create grid
xx3 = np.sort(xx2)
xx3 = np.unique(xx3)
posx_l = xx3[1]
posx_r = np.flipud(xx3)[1]
xx4 = np.arange(posx_l, posx_r, 20)
yy3 = np.sort(yy2)
yy3 = np.unique(yy3)

posy_d = yy3[0]
posy_u = np.flipud(yy3)[1]
yy4 = np.arange(posy_d, posy_u, 10)

for ii in list(range(len(yy4))):
    plt.plot([posx_l, posx_r], [yy4[ii], yy4[ii]], 'r')

for ii in list(range(len(xx4))):
    plt.plot([xx4[ii], xx4[ii]], [posy_d, posy_u], 'b')

plt.show()

plt.savefig('tt1.png')


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
# Calculate the number of East-West rows
interval = 80
depth = 12
h_raw_num = np.floor((max_y - min_y)/(interval + depth))



# Judge whether the point is in the ploygon
coor = np.array([xx2,yy2])
co0r = coor.T[:len(xx2) - 1,:]
# fig = plt.figure()
# axes = fig.add_subplot(1)
# p1 = plt.Polygon(cor)
# axes.add_patch(p1)
# plt.show()
# from shapely import geometry
import shapely.geometry as geometry

def if_inPoly (polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

pt2 = (400, 400)
print(if_inPoly(co0r, pt2))


# %%
# Results of two straight lengths
def closest(mylist, Number):
    answer = Number - mylist - 5
    if all(answer < 0) == True:
        return [0, Number]
    else:
        answer1 = answer[(answer >= 0)]
        answer1 = min(answer1)
        indd = np.where(answer == answer1)
        return [mylist[indd[0]], answer1]


ll = 325  # as example
sche = {}

for ii in np.flipud(len_bulid_total):
    times = ii - len_bulid_total[0]
    ll1 = ll - (ii + 2.5)*2
    build_num = np.floor(ll1/(ii + 5))
    mod_num = np.mod(ll1, (ii + 5))
    if mod_num < min(len_bulid_total):
        sche_name = str(ii)
        sche[sche_name] = [ii,build_num + 2,mod_num]
    else:
        aa = closest(len_bulid_total [:times - 1],mod_num)
        sche_name = str(ii)
        sche[sche_name] = [ii,build_num + 2, mod_num, aa[0], aa[1]]
