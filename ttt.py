
# %%
from numpy.core.function_base import linspace
from shapely import geometry

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)
    
square = np.array([[0,0], [1,0], [1,1], [0,1]]) #多边形坐标
pt1 = (2, 2) #点坐标
pt2 = (0.5, 0.5)
print(if_inPoly(square, pt1))
print(if_inPoly(square, pt2))




#  %% 
import matplotlib.pyplot as plt
import numpy as np
square = np.array([[0,0], [1,0], [1,1], [0,1]])
fig=plt.figure()
axes=fig.add_subplot(1,1,1)
p3=plt.Polygon(square)
axes.add_patch(p3)
plt.show()


# %%
y_max=330

while y_max < 370:
    y_max += 0.5

    # print(y_max)
    ll=y_max-10
    if ll < 300:
        a= y_max
        return a
    #     # break
    # else:
    #     continue

    # %%
    import numpy as np
    y_max=360
    for ii in np.linspace(0.01,40,2000):
        y_max=y_max-ii
        if y_max<350:
            break
    print(y_max)

