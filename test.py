# %%
import numpy as np
import matplotlib.pyplot as plt
xx=np.array([0,300,340,350,10,0])
yy=np.array([340,350,330,10,10,340])
plt.figure(dpi=300,figsize=(4,4))
plt.plot(xx,yy)

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

# 去除最大角
deg1=[]
for ii in list(range(len(slope)-1)):
    # print(ii)
    ang=(slope[ii+1]-slope[ii])/(1+slope[ii+1]*slope[ii])
    deg=deg=np.arctan(ang)* 180 / np.pi
    if deg <0:
        deg=deg+180
    deg1.append(deg)
pos=np.where(deg1==np.max(deg1))

xx1=np.delete(xx,pos[0]+1)
yy1=np.delete(yy,pos[0]+1)
plt.plot(xx1,yy1)

# 退线
max_x=np.max(xx1)
max_y=np.max(yy1)
min_x=np.min(xx1)
min_y=np.min(yy1)
mid_x=(max_x+min_x)/2
mid_y=(max_y+min_y)/2
xx2=[]
yy2=[]
for ii in list(range(len(xx1))):
    # print(ii)
    if xx1[ii]< mid_x:
        xx2.append(xx1[ii]+10)
    else:
        xx2.append(xx1[ii]-10)
    if yy1[ii]< mid_y:
        yy2.append(yy1[ii]+10)
    else:
        yy2.append(yy1[ii]-10)

plt.plot(xx2,yy2)

# 划分简单的网格
xx3=np.sort(xx2)
xx3=np.unique(xx3)
posx_l=xx3[1]
posx_r=np.flipud(xx3)[1]
xx4=np.arange(posx_l,posx_r,20)
yy3=np.sort(yy2)
yy3=np.unique(yy3)
posy_d=yy3[0]
posy_u=np.flipud(yy3)[1]
yy4=np.arange(posy_d,posy_u,10)


for ii in list(range(len(yy4))):
    plt.plot([posx_l,posx_r],[yy4[ii],yy4[ii]],'r')

for ii in list(range(len(xx4))):
    plt.plot([xx4[ii],xx4[ii]],[posy_d,posy_u],'b')

plt.show()

# plt.savefig('tt1.png')

# 计算符合条件的长度组合
len_build=np.array([7,9,11,15])
len_2build=[]
for ii in len_build:
    for jj in len_build:
        len_2build.append(ii+jj)
len_2build=np.unique(len_2build)

len_3build=[]
for ii in len_build:
    for jj in len_build:
        for kk in len_build:
            len_3build.append(ii+jj+kk)
len_3build=np.unique(len_3build)

len_4build=[]
for ii in len_build:
    for jj in len_build:
        for kk in len_build:
            for ll in len_build:
                len_4build.append(ii+jj+kk+ll)
len_4build=np.unique(len_4build)

len_bulid_total=np.hstack((len_2build,len_3build,len_4build))
len_bulid_total=np.sort(len_bulid_total[(len_bulid_total>18) & (len_bulid_total<50)])
len_bulid_total=np.unique(len_bulid_total)

# %%
# 计算东西向能排行数
interval=80
depth=12
h_raw_num=np.floor((max_y-min_y)/(interval+depth))



# %%
# 判断点是否在区域内
coor=np.array([xx2,yy2])
co0r=coor.T[:len(xx2)-1,:]
# fig=plt.figure()
# axes=fig.add_subplot(1)
# p1=plt.Polygon(cor)
# axes.add_patch(p1)
# plt.show()
# from shapely import geometry
import shapely.geometry as geometry

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

pt2 = (400, 400)
print(if_inPoly(co0r, pt2))