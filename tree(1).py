# %%
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely.geometry as geometry
from io import StringIO

# xx=np.array([0,150,250,350, 200,110, 20,0])/2
# yy=np.array([340,460,450,-60,-90 ,-120, 80,340])/2

xx = np.array([100, 350, 650, 800, 300, 510, 150, 100])
yy = np.array([700, 770, 690, 610, 440, 10, 80, 700])
xx = xx/2
yy = yy/2

def polyg(xx,yy):
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
    pg = Polygon(np.genfromtxt(StringIO(ff))) # 生成Polygon实例
    return geometry.Point([xx, yy]).within(pg)

xx1=np.array([100,180,180,100,100])
yy1=np.array([150,150,123,123,150])
xx2=np.array([125,155,155,125,125])
yy2=np.array([75,75,101,101,75])
plt.plot(xx1,yy1)
plt.plot(xx2,yy2)
plt.plot(xx,yy)

for ii in range(len(xx)-1):
    ll=cal_len_edge(xx[ii],yy[ii],xx[ii+1],yy[ii+1])
    num_tree=np.floor(ll/15)
    mod_dis=np.mod(ll,15)
    if mod_dis <= 7.5:
        tree_dis= ll/ num_tree
        ll1=0
        x=xx[ii]
        y=yy[ii]
        while ll1<=ll-mod_dis:
            ll1 += tree_dis
            r=random.uniform(1.5,3)
            theta = np.arange(0, 2*np.pi, 0.01)
            x1 = x + r * np.cos(theta)
            y1 = y + r * np.sin(theta)
            plt.plot(x1,y1, color='r')

            x += 15* (xx[ii+1]-xx[ii])/ll
            y += 15* (yy[ii+1]-yy[ii])/ll

    else:
        ll1=0
        x=xx[ii]
        y=yy[ii]
        while ll1<=ll:
            ll1 += 15
            r=random.uniform(1.5,3)
            theta = np.arange(0, 2*np.pi, 0.01)
            x1 = x + r * np.cos(theta)
            y1 = y+ r * np.sin(theta)
            plt.plot(x1,y1, color='r')

            x += 15* (xx[ii+1]-xx[ii])/ll
            y += 15* (yy[ii+1]-yy[ii])/ll

# plt.show()
# %%

x_max=max(xx)
x_min=min(xx)
y_max=max(yy)
y_min=min(yy)

x_tree=[]
y_tree=[]
while  len(y_tree) <= 50:
    x2=random.uniform(x_min,x_max)
    y2=random.uniform(y_min,y_max)
    if points(x2,y2, polyg(xx,yy)) == True and points(x2, y2, polyg(xx1,yy1)) == False and points(x2, y2, polyg(xx2,yy2)) == False:
         
        y_tree.append(y2)
        x_tree.append(x2)
        r=random.uniform(1.5,3)
        theta = np.arange(0, 2*np.pi, 0.01)
        x1 = x2 + r * np.cos(theta)
        y1 = y2+ r * np.sin(theta)
        plt.plot(x1,y1, color='r')

plt.show()


# %%
outer=polyg(xx,yy)
outer1 = Polygon(np.genfromtxt(StringIO(outer)))
inner=polyg(xx1,yy1)
inner1 = Polygon(np.genfromtxt(StringIO(inner)))
pg = Polygon(outer1,inner1)
plt.plot(xx,yy)
plt.plot(xx1,yy1)


# %%
xxx=np.hstack([xx,xx1])
yyy=np.hstack([yy,yy1])
polyg


# %%
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