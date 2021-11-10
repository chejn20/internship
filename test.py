# %%
import numpy as np
import matplotlib.pyplot as plt
xx=np.array([0,100,140,150,10,0])
yy=np.array([140,150,130,10,10,140])
plt.figure(dpi=300,figsize=(4,4))
plt.plot(xx,yy)
xx_dif=np.diff(xx)
yy_dif=np.diff(yy)
slope=yy_dif/xx_dif
slope=np.append(slope,slope[0])

deg1=[]
for ii in list(range(len(slope)-1)):
    # print(ii)
    ang=(slope[ii+1]-slope[ii])/(1+slope[ii+1]*slope[ii])
    deg=deg=np.arctan(ang)* 180 / np.pi
    if deg <0:
        deg=deg+180
    deg1.append(deg)
pos=np.where(deg1==np.max(deg1))
# xx[pos[0]+1]
xx1=np.delete(xx,pos[0]+1)
yy1=np.delete(yy,pos[0]+1)
plt.plot(xx1,yy1)


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

plt.savefig('tt1.png')