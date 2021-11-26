# %%
import numpy as np
import matplotlib.pyplot as plt
xx=np.array([0,150,250,350,10,0])
yy=np.array([340,340,360,10,10,340])
# yy = np.array([300, 300, 330, 10, 20, 300])
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
# deg1=[]
# for ii in list(range(len(slope)-1)):
#     # print(ii)
#     ang=(slope[ii+1]-slope[ii])/(1+slope[ii+1]*slope[ii])
#     deg=deg=np.arctan(ang)* 180 / np.pi
#     if deg <0:
#         deg=deg+180
#     deg1.append(deg)
# pos=np.where(deg1==np.max(deg1))

# xx1=np.delete(xx,pos[0]+1)
# yy1=np.delete(yy,pos[0]+1)
# plt.plot(xx1,yy1)

# 退线
max_x=np.max(xx)
max_y=np.max(yy)
min_x=np.min(xx)
min_y=np.min(yy)
mid_x=(max_x+min_x)/2
mid_y=(max_y+min_y)/2
xx2=[]
yy2=[]
for ii in list(range(len(xx))):
    # print(ii)
    if xx[ii]< mid_x:
        xx2.append(xx[ii]+10)
    else:
        xx2.append(xx[ii]-10)
    if yy[ii]< mid_y:
        yy2.append(yy[ii]+10)
    else:
        yy2.append(yy[ii]-10)

plt.plot(xx2,yy2)

# 划分简单的网格
# xx3=np.sort(xx2)
# xx3=np.unique(xx3)
# posx_l=xx3[1]
# posx_r=np.flipud(xx3)[1]
# xx4=np.arange(posx_l,posx_r,20)
# yy3=np.sort(yy2)
# yy3=np.unique(yy3)
# posy_d=yy3[0]
# posy_u=np.flipud(yy3)[1]
# yy4=np.arange(posy_d,posy_u,10)


# for ii in list(range(len(yy4))):
#     plt.plot([posx_l,posx_r],[yy4[ii],yy4[ii]],'r')

# for ii in list(range(len(xx4))):
#     plt.plot([xx4[ii],xx4[ii]],[posy_d,posy_u],'b')

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

pt2 = [10,330]
print(if_inPoly(co0r, pt2))

# %%
# 考虑直排两种长度的结果

def closest(mylist, Number):
    answer=Number-5-mylist
    if all(answer<0)==True:
        return [0,Number]
    else:
        answer1=answer[(answer>=0)]
        answer1=min(answer1)
        indd=np.where(answer==answer1)
        return [mylist[indd[0]],answer1]


# ll=325  # 随便定的长度
def arr_ew(ll,ll_build):
    # ll 是东西长度， ll_build 是所有建筑的长度组合
    sche={}

    for ii in np.flipud(ll_build):
        times=ii-ll_build[0]
        if ll>(ii+2.5)*2:
            ll1=ll-(ii+2.5)*2
            build_num=np.floor(ll1/(ii+5))
            mod_num=np.mod(ll1,(ii+5))
            if mod_num<min(ll_build):
                sche_name=str(ii)
                sche[sche_name]=[ii,build_num+2,mod_num,2*2.5+build_num*5+mod_num]
            else:
                aa=closest(ll_build[:times-1],mod_num)
                sche_name=str(ii)
                sche[sche_name]=[ii,build_num+2,mod_num,aa[0],aa[1],2*2.5+(build_num+1)*5+aa[1]]

        if ll< (ii+2.5)*2  and  ll>ii+5:
            ll1=ll-ii
            build_num=np.floor(ll1/(ii+5))
            mod_num=np.mod(ll1,(ii+5))
            if mod_num<min(ll_build):
                sche_name=str(ii)
                sche[sche_name]=[ii,build_num+1,mod_num,2*2.5+build_num*5+mod_num]
            else:
                aa=closest(ll_build[:times-1],mod_num)
                sche_name=str(ii)
                sche[sche_name]=[ii,build_num+1,mod_num,aa[0],aa[1],2*2.5+(build_num+1)*5+aa[1]]

    return sche


# 1112   (作废)
# %%
#计算多边形重心

def core_x_tri(cor_x):
    # 计算三角形重心x及面积
    core_x=np.sum(cor_x)/3
    return [core_x]

def core_y_tri(cor_y):
    # 计算三角形重心y及面积
    core_y=np.sum(cor_y)/3
    return [core_y]

def core_area_tri(cor_x,cor_y):
    # 计算三角形面积
    S=((cor_x[2]-cor_x[0])*(cor_y[1]-cor_y[0])-(cor_x[1]-cor_x[0])*(cor_y[2]-cor_y[0]))/2
    return [S]
# %%

def cut_polyg(arr_x,arr_y):
    x=arr_x
    y=arr_y
    tri_x=[]
    tri_y=[]
    tri_area=[]
    tem_x=x[list(range(0,len(x),2))]
    tem_y=y[list(range(0,len(y),2))]

    for ii in list(range(len(x)-2)):
        if ii % 2==0:
            cor_x_tri=x[ii:3]
            cor_y_tri=y[ii:3]
            tri_x.append(core_x_tri(cor_x_tri))
            tri_y.append(core_y_tri(cor_y_tri))
            tri_area.append(core_area_tri(cor_x_tri,cor_y_tri))

xx=np.array([150,220,300,400,300,150,50,150])
yy=np.array([300,350,280,150,20,15,130,300])
plt.plot(xx,yy)

# %%
slope1=slope_cal(xx2,yy2)
west_slope=slope1[-2]
east_slope=slope1[1]

const1=yy2[0]-west_slope*xx2[0]
const2=yy2[1]-west_slope*xx2[1]
# posi_x_first=abs(np.sort(yy2)[-2]/west_slope)
# 计算东西向能排行数(大致)
# interval=80
# depth=12
# h_raw_num=np.floor((max_y-min_y)/(interval+depth))
yy_raw=np.arange(np.sort(np.unique(yy2))[-2],np.sort(np.unique(yy2))[0],-(interval+depth))
if yy_raw[-1]<depth:
    np.delete(yy_raw,-1)
posi_x_l=(yy_raw-const1)/west_slope
posi_x_r=(yy_raw-const2)/west_slope

plt.plot(xx2,yy2)
for ii in list(range(len(yy_raw))):
    plt.plot([posi_x_l[ii],posi_x_r[ii]],[yy_raw[ii],yy_raw[ii]],'r')


# %%
# 计算东西向能排行数(大致)
interval=80
depth=12
h_raw_num=np.floor((max_y-min_y)/(interval+depth))

# %% 1113
north_west_x=[]
north_west_y=[]
for ii in list(range(len(xx)-1)):
    # print(ii)
    point_x=xx[ii]
    if point_x<mid_x:
        north_west_x.append(point_x)
        north_west_y.append(yy[ii])
north_west_x1=[]
north_west_y1=[]
if len(north_west_x) != 0:
    for ii in list(range(len(north_west_y))):
        if north_west_y[ii]>mid_y:
            north_west_y1.append(north_west_y[ii])
            north_west_x1.append(north_west_x[ii])
if len(north_west_x1)==1:
    north_west_x_final=north_west_x1
    north_west_y_final=north_west_y1

if len(north_west_x1)>1:
    slope_mid_point=[]
    length_mid_point=[]
    for ii in list(range(len(north_west_x1))):
        slope_mid_point.append((north_west_y1[ii]-mid_y)/(north_west_x1-mid_x))
        length_mid_point.append(np.sqrt((north_west_y1[ii]-mid_y)**2+(north_west_x1-mid_x)**2))
    slope_mid_point_1=np.abs(np.array(slope_mid_point)+1)
    posi_slope_min=np.where(slope_mid_point_1==np.min(slope_mid_point_1))
    if length_mid_point[posi_slope_min[0]]== np.max(length_mid_point):
        north_west_x_final=north_west_x1[posi_slope_min[0]]
        north_west_y_final=north_west_y1[posi_slope_min[0]]
    else:
        north_west_x_final=north_west_x1[np.where(slope_mid_point==np.min(slope_mid_point))[0]]
        north_west_x_final=north_west_x1[np.where(slope_mid_point==np.min(slope_mid_point))[0]]

if len(north_west_x)==0:
    north_east_x=[]
    north_east_y=[]
    for ii in list(range(len(xx)-1)):
        # print(ii)
        point_x=xx[ii]
        if point_x>mid_x:
            north_east_x.append(point_x)
            north_east_y.append(yy[ii])
    north_east_x1=[]
    north_east_y1=[]

    for ii in list(range(len(north_east_y))):
        if north_east_y[ii]>mid_y:
            north_east_y1.append(north_east_y[ii])
            north_east_x1.append(north_east_x[ii])
            
    if len(north_east_x1)==1:
        north_east_x_final=north_east_x1
        north_east_y_final=north_east_y1

    if len(north_east_x1)>1:
        slope_mid_point=[]
        length_mid_point=[]
        for ii in list(range(len(north_east_x1))):
            slope_mid_point.append((north_east_y1[ii]-mid_y)/(north_east_x1-mid_x))
            length_mid_point.append(np.sqrt((north_east_y1[ii]-mid_y)**2+(north_east_x1-mid_x)**2))
        slope_mid_point_1=np.abs(np.array(slope_mid_point)-1)
        posi_slope_min=np.where(slope_mid_point_1==np.min(slope_mid_point_1))
        if length_mid_point[posi_slope_min[0]]== np.max(length_mid_point):
            north_east_x_final=north_east_x1[posi_slope_min[0]]
            north_east_y_final=north_east_y1[posi_slope_min[0]]
        else:
            north_east_x_final=north_east_x1[np.where(slope_mid_point==np.max(slope_mid_point))[0]]
            north_east_x_final=north_east_x1[np.where(slope_mid_point==np.max(slope_mid_point))[0]]

# %%
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

    flag = -1
    for i in range(1, xx.shape[0]):
        if x < max(xx[i], xx[i - 1]) and (min(yy[i], yy[i - 1]) < y < max(yy[i], yy[i - 1])):
            flag = -1 * flag
        elif yy[i] == y and yy[i - 1] == y and x < max(xx[i], xx[i - 1]):
            continue
        elif y == max(yy[i], yy[i - 1]) and x < max(xx[i], xx[i - 1]):
            flag = -1 * flag
        elif y == min(yy[i], yy[i - 1]) and x < max(xx[i], xx[i - 1]):
            continue
    if flag == 1:
        return True
    return False

isInPloygon(np.array(xx2),np.array(yy2),10,330)

# %%

def calculation_SN(xx, yy, ymax):
    # xx 是退线之后点横坐标
    # yy 是退线之后点纵坐纵
    y_max=ymax
    # y_min=np.min(yy2)
    dya=np.array(yy)-y_max

    intersec_x=[]
    intersec_y=[]
    for ii in list(range(len(dya)-1)):
        mult=dya[ii]*dya[ii+1]
        if mult<0:
            intersec_x.append(xx[ii])
            intersec_x.append(xx[ii+1])
            intersec_y.append(yy[ii])
            intersec_y.append(yy[ii+1])

    slope_intersec=[]
    const_intersec=[]

    for ii in list(range(len(intersec_x))):
        if ii % 2 != 0:
            temp=(intersec_y[ii]-intersec_y[ii-1])/(intersec_x[ii]-intersec_x[ii-1])
            slope_intersec.append(temp) 
            temp1=intersec_y[ii]-temp*intersec_x[ii]
            const_intersec.append(temp1)

    if intersec_x[1] <= intersec_x[2]:     #第一个点在最高点左侧


        if slope_intersec[0] > 0 and slope_intersec[1] < 0:    # 左边斜率大于0 右边斜率小于0
            intersec_posi_x1=(y_max-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] > 0 and slope_intersec[1] > 0:    # 左边斜率大于0 右边斜率大于0
            intersec_posi_x1=(y_max-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-12-const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] < 0:    # 左边斜率小于0 右边斜率小于0
            intersec_posi_x1=(y_max-12-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] > 0:    # 左边斜率小于0 右边斜率小大于0
            intersec_posi_x1=(y_max-12-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-12-const_intersec[1])/slope_intersec[1]
        return intersec_posi_x1, intersec_posi_x2

    elif intersec_x[1] >= intersec_x[2]:     #第一个点在最高点右侧


        if slope_intersec[0] > 0 and slope_intersec[1] < 0:    # 左边斜率大于0 右边斜率小于0
            intersec_posi_x1=(y_max-12-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-12-const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] > 0 and slope_intersec[1] > 0:    # 左边斜率大于0 右边斜率大于0
            intersec_posi_x1=(y_max-12-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] < 0:    # 左边斜率小于0 右边斜率小于0
            intersec_posi_x1=(y_max-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-12-const_intersec[1])/slope_intersec[1]

        elif slope_intersec[0] < 0 and slope_intersec[1] > 0:    # 左边斜率小于0 右边斜率小大于0
            intersec_posi_x1=(y_max-const_intersec[0])/slope_intersec[0]
            intersec_posi_x2=(y_max-const_intersec[1])/slope_intersec[1]

        return intersec_posi_x2, intersec_posi_x1

# 计算东西向能排行数(大致)
interval=80
depth=12
h_raw_num=np.floor((max_y-min_y)/(interval+depth))
print(h_raw_num)
plt.plot(xx2,yy2)
y_max=np.max(yy2)
y_min=np.min(yy2)


for i in list(range(int(h_raw_num+2))):
    if y_max-92-12<y_min:
        break
    
    # intersec_posi_x1_total=[]
    # intersec_posi_x2_total=[]

    if i==0:
        
        y_max=y_max-6
        intersec_posi_x1,intersec_posi_x2=calculation_SN(xx2,yy2,y_max)

        # intersec_posi_x1_total.append(intersec_posi_x1)
        # intersec_posi_x2_total.append(intersec_posi_x2)
    
        ll1=np.abs(intersec_posi_x2-intersec_posi_x1)


        plt.plot([intersec_posi_x1,intersec_posi_x2],[y_max, y_max])

        temp=arr_ew(ll1,len_bulid_total)

        mod_aera=[]
        for ii in len_bulid_total:
            if ii+5>ll1:
                break
            if len(temp[str(ii)])==4:
                mod_aera.append(temp[str(ii)][3])
            if len(temp[str(ii)])==6:
                mod_aera.append(temp[str(ii)][5])

        min_mod=np.where(mod_aera==min(mod_aera))

        num_build=temp[str(len_bulid_total[min_mod[0][0]])][1]
        for jj in list(range(int(num_build))):
            x_build_start=intersec_posi_x1+jj*(5+temp[str(len_bulid_total[min_mod[0][0]])][0])
            x_build_second=x_build_start+temp[str(len_bulid_total[min_mod[0][0]])][0]
            x_build=np.hstack((x_build_start,x_build_second))
            x_build=np.hstack((x_build, np.flipud(x_build)))
            x_build=np.append(x_build, x_build_start)

            y_build=np.array([y_max, y_max, y_max-12, y_max-12, y_max])

            plt.plot(x_build,y_build)


        print('排楼方法为：'+str(temp[str(len_bulid_total[min_mod[0][0]])]))



    else:
        y_max=y_max-92

        intersec_posi_x1,intersec_posi_x2=calculation_SN(xx2,yy2,y_max)

        # intersec_posi_x1_total.append(intersec_posi_x1)
        # intersec_posi_x2_total.append(intersec_posi_x2)
        

        ll1=np.abs(intersec_posi_x2-intersec_posi_x1)
        plt.plot([intersec_posi_x1,intersec_posi_x2],[y_max, y_max])

        temp=arr_ew(ll1,len_bulid_total)

        mod_aera=[]
        for ii in len_bulid_total:
            if ii+5>ll1:
                break
            if len(temp[str(ii)])==4:
                mod_aera.append(temp[str(ii)][3])
            if len(temp[str(ii)])==6:
                mod_aera.append(temp[str(ii)][5])

        min_mod=np.where(mod_aera==min(mod_aera))

        if len(temp[str(len_bulid_total[min_mod[0][0]])]) ==4:

            num_build=temp[str(len_bulid_total[min_mod[0][0]])][1]
            for jj in list(range(int(num_build))):
                x_build_start=intersec_posi_x1+jj*(5+temp[str(len_bulid_total[min_mod[0][0]])][0])
                x_build_second=x_build_start+temp[str(len_bulid_total[min_mod[0][0]])][0]
                x_build=np.hstack((x_build_start,x_build_second))
                x_build=np.hstack((x_build, np.flipud(x_build)))
                x_build=np.append(x_build, x_build_start)

                y_build=np.array([y_max, y_max, y_max-12, y_max-12, y_max])

                plt.plot(x_build,y_build)
        
        elif len(temp[str(len_bulid_total[min_mod[0][0]])]) ==6:
            num_build=temp[str(len_bulid_total[min_mod[0][0]])][1]+1
            for jj in list(range(int(num_build))):
                x_build_start=intersec_posi_x1+jj*(5+temp[str(len_bulid_total[min_mod[0][0]])][0])
                if jj < num_build:
                    x_build_second=x_build_start+temp[str(len_bulid_total[min_mod[0][0]])][0]
                elif jj == num_build:
                    x_build_second=x_build_start+temp[str(len_bulid_total[min_mod[0][0]])][3]
                x_build=np.hstack((x_build_start,x_build_second))
                x_build=np.hstack((x_build, np.flipud(x_build)))
                x_build=np.append(x_build, x_build_start)

                y_build=np.array([y_max, y_max, y_max-12, y_max-12, y_max])

                plt.plot(x_build,y_build)



        print('排楼方法为：'+str(temp[str(len_bulid_total[min_mod[0][0]])]))



    






