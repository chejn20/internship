range(len(y_up)-1):
#     if ii == 0:
#         start_point_x = x_lengh[0]
#         # x_1=start_point_x
#         second_point_x = x_lengh[1]
#         x_build = np.hstack((start_point_x, second_point_x))
#         x_build = np.hstack((x_build, np.flipud(x_build)))
#         x_build = np.append(x_build, start_point_x)
#         y_build = np.array([y_up[0], y_up[0], y_up[0]-12, y_up[0]-12, y_up[0]])
#         plt.plot(x_build,y_build)

#     else:
#         if y_up[ii] < y_up[ii+1]:
#             y_max = y_up[ii]
#             if y_down[ii] < y_down[ii+1]:
#                 y_min= y_down[ii+1]
#             elif y_down[ii] > y_down[ii+1]:
#                 y_min= y_down[ii]
#                 len_NS= y_max - y_min 


#         elif y_up[ii] > y_up[ii+1]:
#             y_max = y_up[ii+1]
#             if y_down[ii] < y_down[ii+1]:
#                 y_min= y_down[ii+1]
#             elif y_down[ii] > y_down[ii+1]:
#                 y_min= y_down[ii]
#                 len_NS= y_max - y_min 

#         build_num_NS=np.floor(len_NS/92)
#         mod_NS=np.mod(len_NS,92)
#         if mod_NS > 12:
#             for jj in range(int(build_num_NS)+1):
#                 if jj != range(int(build_num_NS)+1)[-1]:
#                     start_point_x = x_lengh[ii]
#                     # x_1=start_point_x
#                     second_point_x = x_lengh[ii+1]
#                     x_build = np.hstack((start_point_x, second_point_x))
#                     x_build = np.hstack((x_build, np.flipud(x_build)))
#                     x_build = np.append(x_build, start_point_x)
#                     y_build = np.array([y_max-jj*92, y_max-jj*92, y_max-12-jj*92, y_max-12-jj*92, y_max-jj*92])
#                     plt.plot(x_build,y_build)
#                 elif jj == range(int(build_num_NS)+1)[-1]:
#                     start_point_x = x_lengh[ii]
#                     # x_1=start_point_x
#                     second_point_x = x_lengh[ii+1]
#                     x_build = np.hstack((start_point_x, second_point_x))
#                     x_build = np.hstack((x_build, np.flipud(x_build)))
#                     x_build = np.append(x_build, start_point_x)
#                     y_build = np.array([y_min+12, y_min+12, y_min, y_min, y_min+12])
#                     plt.plot(x_build,y_build)