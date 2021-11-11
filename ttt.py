
# %%
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