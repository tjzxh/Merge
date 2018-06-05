import matplotlib.pyplot as plt
from process_log import process_pcmap
import process_log
import numpy as np
import cv2 as cv
from scipy.linalg import solve
import json
import math
from sklearn.metrics.pairwise import cosine_similarity

gps = 0
vslam = 0
all_data0 = process_pcmap("left_out.pcmap", vslam, gps)
all_x0 = all_data0[:, 0]
all_y0 = all_data0[:, 1]
all_data1 = process_pcmap("right_out.pcmap", vslam, gps)
all_x1 = all_data1[:, 0]
all_y1 = all_data1[:, 1]
width = (max(max(all_x0), max(all_x1)) - min(min(all_x0), min(all_x1))) / 10
height = (max(max(all_y0), max(all_y1)) - min(min(all_y0), min(all_y1))) / 10
dpi = 100
figure1 = plt.figure(figsize=(width, height), dpi=dpi)
# figure1 = plt.figure()
ax1 = plt.gca()  # figure1.add_axes([0, 0, width, height])
ax1.set_aspect(1)
# figure1.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.axis('off')
# 设置坐标轴范围
plt.xlim((min(min(all_x0), min(all_x1)) - 1, max(max(all_x0), max(all_x1)) + 1))
plt.ylim((min(min(all_y0), min(all_y1)) - 1, max(max(all_y0), max(all_y1)) + 1))
ax1.plot(all_x0, all_y0, c='r', linewidth=10, markersize=1)
figure1.savefig("77.png", dpi=dpi, transparent=True)

figure2 = plt.figure(figsize=(width, height), dpi=dpi)
# figure2 = plt.figure()
plt.axis('off')
# figure2.tight_layout(pad=0, w_pad=0, h_pad=0)
ax2 = plt.gca()
ax2.set_aspect(1)
plt.xlim((min(min(all_x0), min(all_x1)) - 1, max(max(all_x0), max(all_x1)) + 1))
plt.ylim((min(min(all_y0), min(all_y1)) - 1, max(max(all_y0), max(all_y1)) + 1))
ax2.plot(all_x1, all_y1, c="r", linewidth=10, markersize=1)
figure2.savefig("88.png", dpi=dpi, transparent=True)
plt.show()

# convert real coordinate into picture
left = min(min(all_x0), min(all_x1))
right = max(max(all_x0), max(all_x1))
top = max(max(all_y0), max(all_y1))
bottom = min(min(all_y0), min(all_y1))
# read picture
img1 = cv.imread('77.png', 0)
img2 = cv.imread('88.png', 0)
ret1, grey1 = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)
ret2, grey2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
all = cv.addWeighted(grey1, 0.5, grey2, 0.5, 0)
overlap = np.where(all == 0)
# cluster the points
all_label = [0]
label = 0
for k in range(len(overlap[0]) - 1):
    if (overlap[0][k + 1] - overlap[0][k]) + (overlap[1][k + 1] - overlap[1][k]) <= 50:
        all_label.append(label)
    else:
        label += 1
        all_label.append(label)
# print(min(all_label), max(all_label))
img = np.where(all != 255)
img_y = img[0]
img_x = img[1]
img_top = min(img_y)
img_bottom = max(img_y)
img_right = max(img_x)
img_left = min(img_x)
# obtain the convert parameters
a = np.array([[img_left, 1, 0, 0], [img_right, 1, 0, 0], [0, 0, img_top, 1], [0, 0, img_bottom, 1]])
b = np.array([left, right, top, bottom])
coef = solve(a, b)
all_overlap = []
all_x = []
all_y = []
for i in range(len(overlap[0])):
    y_pic = overlap[0][i]
    x_pic = overlap[1][i]
    x = coef[0] * x_pic + coef[1]
    y = coef[2] * y_pic + coef[3]
    if [x, y, all_label[i]] not in all_overlap:
        all_overlap.append([x, y, all_label[i]])
    all_x.append(x)
    all_y.append(y)
# all the overlap points
plt.figure()
plt.plot(all_x0, all_y0, c='b')
plt.plot(all_x1, all_y1, c='b')
plt.scatter(all_x, all_y, c='r', s=1)
plt.show()
# _, loc0 = process_log.find_neighbor(all_data0, all_overlap)
# _, loc1 = process_log.find_neighbor(all_data1, all_overlap)
all_overlap = np.array(all_overlap)
overlap_od0 = []
overlap_od1 = []
for j in range(min(all_label), max(all_label) + 1):
    same_class_id = np.where(all_overlap[:, 2] == j)
    same_class = all_overlap[same_class_id]
    loc0 = process_log.find_neighbor(all_data0, same_class)
    loc1 = process_log.find_neighbor(all_data1, same_class)
    if (loc0[0] - loc0[-1]) * (loc1[0] - loc1[-1]) < 0:
        continue
    else:
        overlap_od0.append([min(loc0), max(loc0)])
        overlap_od1.append([min(loc1), max(loc1)])
# overlap seg
fig = plt.figure()

overlap_data0 = all_data0[overlap_od0[0][0]:overlap_od0[0][1]]
overlap_data1 = all_data1[overlap_od1[0][0]:overlap_od1[0][1]]
plt.subplot(121)
plt.plot(all_x0, all_y0, c='b')
plt.scatter(overlap_data0[:, 0], overlap_data0[:, 1], c='r')
plt.subplot(122)
plt.plot(all_x1, all_y1, c='b')
plt.scatter(overlap_data1[:, 0], overlap_data1[:, 1], c='lime')
plt.show()

# step7 Generate a whole json file for hmap
if gps == 0:
    is_coordinate_gps = 0
else:
    is_coordinate_gps = 1
hmap = {"debug_info": {"lat_bias": 0, "lng_bias": 0}, "is_coordinate_gps": is_coordinate_gps,
        "lane_switch_set": [],
        "node_set": [], "segment_set": []}
id4node = 0
id4seg = 0
width = 3
max_vel_str = 20
max_vel_cur = 15
gps = 0
autovel = 0
weight = [1, 0, 0, 0]
is_local = 0
len0 = len(all_data0)
id4connect0 = 0
id4connect1 = 0
# traverse all the overlap seg
# if the original point of overlap is the first point of trajectory
for d in range(len(overlap_od0)):
    if d == 0:
        if overlap_od0[0][0] == 0 and overlap_od1[0][0] == 0:
            # the first seg is the overlap seg
            overlap0 = all_data0[overlap_od0[d][0]:overlap_od0[d][1]]
            hmap, id4node, id4seg = process_log.single_dump(overlap0, hmap, width, max_vel_str, max_vel_cur, gps,
                                                            autovel, weight, is_local, id4node, id4seg)
        else:
            if overlap_od0[0][0] != 0:
                vector0 = [all_data0[overlap_od0[d][0] - 1][0] - all_data0[overlap_od0[d][0] - 2][0],
                           all_data0[overlap_od0[d][0] - 1][1] - all_data0[overlap_od0[d][0] - 2][1]]
            if overlap_od1[0][0] != 0:
                vector1 = [all_data1[overlap_od1[d][0] - 1][0] - all_data1[overlap_od1[d][0] - 2][0],
                           all_data1[overlap_od1[d][0] - 1][1] - all_data1[overlap_od1[d][0] - 2][1]]
            # find the real start of overlap
            ov_st = overlap_od0[d][0]
            vector4connect = [all_data0[ov_st][0] - all_data0[ov_st - 1][0],
                              all_data0[ov_st][1] - all_data0[ov_st - 1][1]]
            if overlap_od0[0][0] != 0:
                cos0 = cosine_similarity([vector0], [vector4connect])
                while cos0 < math.cos(10 * math.pi / 180):
                    ov_st += 1
                    vector4connect = [all_data0[ov_st][0] - all_data0[ov_st - 1][0],
                                      all_data0[ov_st][1] - all_data0[ov_st - 1][1]]
                    cos0 = cosine_similarity([vector0], [vector4connect])
            if overlap_od1[0][0] != 0:
                cos1 = cosine_similarity([vector1], [vector4connect])
                while cos1 < math.cos(10 * math.pi / 180):
                    ov_st += 1
                    vector4connect = [all_data0[ov_st][0] - all_data0[ov_st - 1][0],
                                      all_data0[ov_st][1] - all_data0[ov_st - 1][1]]
                    cos1 = cosine_similarity([vector1], [vector4connect])
            # find another start of overlap in another trajectory
            ov_st1 = process_log.find_neighbor(all_data1, np.array([all_data0[ov_st][0], all_data0[ov_st][1]]).reshape(1, 2))
            ov_st1 = ov_st1[0]
            last1 = ov_st1
            last0 = ov_st
            while process_log.distance(all_data1[last1][:2], all_data1[ov_st1][:2]) < 0.5:
                last1 -= 1
            if ov_st1 < overlap_od1[d][0]:
                ov_st1 = overlap_od1[d][0]
            while process_log.distance(all_data0[last0][:2], all_data0[ov_st][:2]) < 0.5:
                last0 -= 1
            if overlap_od0[0][0] != 0:
                first_data0 = all_data0[:last0]
                hmap, id4node, id4seg = process_log.single_dump(first_data0, hmap, width, max_vel_str, max_vel_cur, gps,
                                                                autovel,
                                                                weight, is_local, id4node, id4seg)
                id4connect0 = id4node - 1

            if overlap_od1[0][0] != 0:
                first_data1 = all_data1[:last1]
                hmap, id4node, id4seg = process_log.single_dump(first_data1, hmap, width, max_vel_str, max_vel_cur, gps,
                                                                autovel,
                                                                weight, is_local, id4node, id4seg)
                id4connect1 = id4node - 1
            if overlap_od0[0][0] != 0:
                first_connect_seg0 = {"id": id4seg, "lane_list": [
                    {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
                     "name": "Path" + str(id4seg),
                     "node_list": [id4connect0, id4node], "right_line_type": 1,
                     "seg_id": id4seg}],
                                      "name": "seg" + str(id4seg)}
                id4seg += 1
                hmap["segment_set"].append(first_connect_seg0)
            if overlap_od1[0][0] != 0:
                first_connect_seg1 = {"id": id4seg, "lane_list": [
                    {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
                     "name": "Path" + str(id4seg),
                     "node_list": [id4connect1, id4node], "right_line_type": 1,
                     "seg_id": id4seg}], "name": "seg" + str(id4seg)}
                id4seg += 1
                hmap["segment_set"].append(first_connect_seg1)

            overlap0 = all_data0[ov_st:overlap_od0[d][1]]
            hmap, id4node, id4seg = process_log.single_dump(overlap0, hmap, width, max_vel_str, max_vel_cur, gps,
                                                            autovel, weight, is_local, id4node, id4seg)
    elif d == len(overlap_od0) - 1:
        id4final = id4node - 1
        if overlap_od0[-1][1] != len(all_data0) - 1:
            final_connect_seg0 = {"id": id4seg, "lane_list": [
                {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
                 "name": "Path" + str(id4seg),
                 "node_list": [id4final, id4node], "right_line_type": 1,
                 "seg_id": id4seg}], "name": "seg" + str(id4seg)}
            id4seg += 1
            hmap["segment_set"].append(final_connect_seg0)
            final_data0 = all_data0[overlap_od0[d][1]:]
            hmap, id4node, id4seg = process_log.single_dump(final_data0, hmap, width, max_vel_str, max_vel_cur, gps,
                                                            autovel, weight, is_local, id4node, id4seg)
        if overlap_od1[-1][1] != len(all_data1) - 1:
            final_connect_seg1 = {"id": id4seg, "lane_list": [
                {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
                 "name": "Path" + str(id4seg),
                 "node_list": [id4final, id4node], "right_line_type": 1,
                 "seg_id": id4seg}], "name": "seg" + str(id4seg)}
            id4seg += 1
            hmap["segment_set"].append(final_connect_seg1)
            final_data1 = all_data1[overlap_od1[d][1]:]
            hmap, id4node, id4seg = process_log.single_dump(final_data1, hmap, width, max_vel_str, max_vel_cur, gps,
                                                            autovel, weight, is_local, id4node, id4seg)
    else:
        id4connect_f = id4node - 1
        data0 = all_data0[overlap_od0[d - 1][1]:overlap_od0[d][0]]
        front_connect_seg0 = {"id": id4seg, "lane_list": [
            {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
             "name": "Path" + str(id4seg),
             "node_list": [id4connect_f, id4node], "right_line_type": 1,
             "seg_id": id4seg}],
                              "name": "seg" + str(id4seg)}
        id4seg += 1
        hmap["segment_set"].append(front_connect_seg0)
        hmap, id4node, id4seg = process_log.single_dump(data0, hmap, width, max_vel_str, max_vel_cur, gps,
                                                        autovel, weight, is_local, id4node, id4seg)
        id4connect_b0 = id4node - 1
        data1 = all_data1[overlap_od1[d - 1][1]:overlap_od1[d][0]]
        front_connect_seg1 = {"id": id4seg, "lane_list": [
            {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
             "name": "Path" + str(id4seg),
             "node_list": [id4connect_f, id4node], "right_line_type": 1,
             "seg_id": id4seg}],
                              "name": "seg" + str(id4seg)}
        id4seg += 1
        hmap["segment_set"].append(front_connect_seg1)
        hmap, id4node, id4seg = process_log.single_dump(data1, hmap, width, max_vel_str, max_vel_cur, gps,
                                                        autovel, weight, is_local, id4node, id4seg)
        id4connect_b1 = id4node - 1
        behind_connect_seg0 = {"id": id4seg, "lane_list": [
            {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
             "name": "Path" + str(id4seg),
             "node_list": [id4connect_b0, id4node], "right_line_type": 1,
             "seg_id": id4seg}],
                               "name": "seg" + str(id4seg)}
        id4seg += 1
        hmap["segment_set"].append(behind_connect_seg0)
        behind_connect_seg1 = {"id": id4seg, "lane_list": [
            {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
             "name": "Path" + str(id4seg),
             "node_list": [id4connect_b1, id4node], "right_line_type": 1,
             "seg_id": id4seg}],
                               "name": "seg" + str(id4seg)}
        id4seg += 1
        hmap["segment_set"].append(behind_connect_seg1)
        overlap = all_data0[overlap_od0[d][0]:overlap_od0[d][1]]
        hmap, id4node, id4seg = process_log.single_dump(overlap, hmap, width, max_vel_str, max_vel_cur, gps,
                                                        autovel, weight, is_local, id4node, id4seg)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(MyEncoder, self).default(obj)


with open("0604" + '.hmap', 'w') as f1:
    json.dump(hmap, f1, indent=4, cls=MyEncoder)
