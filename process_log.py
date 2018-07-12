import numpy as np
import re
from scipy.spatial import cKDTree
import utm
from sklearn.cluster import *
from scipy.stats import mode
import math
import os
from rdp import rdp
from scipy.spatial.distance import squareform, pdist


def delete_near_point(all_point, min_dis):
    point = list(all_point[:, 0:2])
    all_point_list = list(all_point[:])
    delete_id = []
    dele = []
    dis = squareform(pdist(np.array(point), 'euclidean'))
    a = np.where(dis < min_dis)
    for i in range(len(a[0])):
        b = [a[0][i], a[1][i]]
        c = [a[1][i], a[0][i]]
        if b in delete_id or c in delete_id or b == c or a[1][i] - a[0][i] != 1:
            pass
        else:
            delete_id.append(b)
            dele.append(list(all_point_list[b[1]]))
    all_point_list = list(map(list, all_point_list))
    for j in dele:
        all_point_list.remove(j)
    return all_point_list


def concat_stright(slope, all_point):
    key_point = []
    origin_point = all_point[0]
    key_point.append(list(origin_point))
    # print(len(slope)//2)
    for k in range(0, len(slope) // 2):
        origin_of_class = all_point[2 * k]
        middle_of_class = all_point[2 * k + 1]
        destination_of_class = all_point[2 * k + 2]
        vector1 = [1, slope[2 * k]]
        vector2 = [1, slope[2 * k + 1]]
        # simi_cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        simi_cos = (1 + slope[2 * k] * slope[2 * k + 1]) / math.sqrt(1 + slope[2 * k] ** 2) / math.sqrt(
            1 + slope[2 * k + 1] ** 2)
        if simi_cos > math.cos(10 * math.pi / 180):
            key_point.append(list(destination_of_class))
        elif simi_cos < math.cos(20 * math.pi / 180):
            key_point.append(list(middle_of_class))
            key_point.append(list(destination_of_class))
        else:
            key_point.append(list(middle_of_class))
            key_point.append(list(destination_of_class))

    if len(slope) % 2 == 1:
        key_point.append(list(all_point[len(slope) - 1]))
        key_point.append(list(all_point[len(slope)]))
    return key_point


def calculate_slope(all_point):
    slope = []
    for j in range(len(all_point) - 1):
        point0 = all_point[j]
        point1 = all_point[j + 1]
        point0_x = float(point0[0])
        point0_y = float(point0[1])
        point1_x = float(point1[0])
        point1_y = float(point1[1])
        if point0_x == point1_x:
            class_slope = 10000
        else:
            class_slope = (point1_y - point0_y) / (point1_x - point0_x)
        slope.append(class_slope)
    return slope


def vector_cos(vector1, vector2):
    cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cos


def process_pcmap(pcmap, vslam, gps):
    if os.path.exists(pcmap) is False:
        print("We can't find the file")
        return False
    else:
        pass
    pcmap_flag = 1
    all_data = []
    all_utm = []
    all_utm_time = []
    all_utm_time_walk = []

    with open(pcmap, 'r') as f:
        for line in f:
            line = line.strip('\n')
            every_line = list(map(float, re.split('[, ]', line)))
            if np.size(every_line) == 1:
                pcmap_flag = every_line[0]
            else:
                all_data.append(every_line)
    # Judge again for pcmap flag
    if pcmap_flag == 0:
        gps = 0
    # Start:Get latitude,longitude,timestamp and height from all kinds of Data
    all_data = np.array(all_data)
    if vslam == 1:
        longi = all_data[:, 1]
        lati = all_data[:, 2]
        time = all_data[:, 0]
        height = all_data[:, 3]
        # elif pcmap_flag == 0:
        # longi = all_data[:, 0]
        # lati = all_data[:, 1]
        # time = list(range(len(longi)))
        # height = all_data[:, 2]
    else:
        longi = all_data[:, 0]
        lati = all_data[:, 1]
        height = all_data[:, 2]
        # time = list(range(len(longi)))
        time = all_data[:, 5]

    if vslam == 1:
        walking_speed = 4 / 66
    else:
        walking_speed = 5

    # step0 process original data

    # step0-0 read original log and delete repetition
    # Local Coordinate or GPS
    all_znum = []
    force = 0
    if gps == 0:
        for i in range(len(longi)):
            u = [longi[i], lati[i]]
            xyz = [u[0], u[1], height[i]]
            if xyz in all_utm:
                all_utm.remove(xyz)
            else:
                all_utm.append(xyz)
                u_time = [u[0], u[1], height[i], time[i]]
                u_time_walk = [u[0], u[1], height[i], time[i] * walking_speed]
                # print(u_time_walk)
                all_utm_time.append(u_time)
                all_utm_time_walk.append(u_time_walk)
                # print(all_utm_time_walk)
    else:
        # whether in the same zone
        for h in range(len(longi)):
            unit = utm.from_latlon(lati[h], longi[h])
            zone_number = unit[2]
            all_znum.append(zone_number)
        if all_znum == [all_znum[0]] * len(all_znum):
            force = 0
        else:
            force = mode(all_znum)
        for i in range(len(longi)):
            if force == 0:
                u = utm.from_latlon(lati[i], longi[i])
            else:
                u = utm.from_latlon(lati[i], longi[i], force_zone_number=force)
            u = list(u)
            all_utm.append(u[0:2])
            if u in all_utm:
                all_utm.remove(u)
            u_time = [u[0], u[1], height[i], time[i]]
            all_utm_time.append(u_time)
    all_utm_time = np.array(all_utm_time)
    # arg = np.argsort(all_utm_time[:, 3])
    # all_point_final = all_utm_time[list(arg)]
    all_point_final = all_utm_time
    return all_point_final


def extract_point(all_utm_time):
    all_pure_utm = all_utm_time[:, :3]
    mask = rdp(all_pure_utm, epsilon=0.02, algo="iter", return_mask=True)
    all_point = all_utm_time[mask]
    return all_point


def find_neighbor(all_point, overlap):
    all_pure_xy = all_point[:, :2]
    tree = cKDTree(all_pure_xy)
    dis, loc = tree.query(overlap[:, :2], k=1)
    return dis, loc


def cluster4overlap(all_point, overlap_loc):
    speed = 5
    all_overlap_od = []
    brc = Birch(threshold=1.5, n_clusters=None, branching_factor=10000)
    overlap = all_point[overlap_loc]
    overlap[:, -1] = overlap[:, -1] * speed
    brc.fit(overlap)
    label = brc.labels_
    for i in list(set(label)):
        same_class = np.where(label == i)
        same_class_id = overlap_loc[same_class[0]]
        min_overlap4id = np.where(same_class_id == min(same_class_id))
        max_overlap4id = np.where(same_class_id == max(same_class_id))
        min_overlap_id = same_class_id[min_overlap4id]
        max_overlap_id = same_class_id[max_overlap4id]
        single_overlap_od = [min_overlap_id[0], max_overlap_id[0]]
        all_overlap_od.append(single_overlap_od)
    return all_overlap_od


def cluster4overlap1(all_point0, all_point1, overlap):
    all_overlap_od0 = []
    all_overlap_od1 = []
    brc = Birch(threshold=1.5, n_clusters=None, branching_factor=10000)
    brc.fit(overlap)
    label = brc.labels_
    for i in list(set(label)):
        same_class_id = np.where(label == i)
        overlap = np.array(overlap)
        same_class = overlap[same_class_id[0]]
        loc0 = find_neighbor(all_point0, same_class)
        loc1 = find_neighbor(all_point1, same_class)
        single_overlap_od0 = [min(loc0), max(loc0)]
        single_overlap_od1 = [min(loc1), max(loc1)]
        all_overlap_od0.append(single_overlap_od0)
        all_overlap_od1.append(single_overlap_od1)
    return all_overlap_od0, all_overlap_od1


def concat_overlap(all_overlap_od, all_point):
    indepent = 0
    if len(all_overlap_od) == 1:
        return all_overlap_od
    else:
        new_overlap_od = []
        for i in range(0, len(all_overlap_od) // 2, 2):
            ori1 = all_overlap_od[i][0]
            des1 = all_overlap_od[i][1]
            ori2 = all_overlap_od[i + 1][0]
            des2 = all_overlap_od[i + 1][1]
            if ori2 <= des1 or distance(all_point[ori2], all_point[des1]) <= 5:
                if [min(ori1, ori2), max(des1, des2)] not in new_overlap_od:
                    new_overlap_od.append([min(ori1, ori2), max(des1, des2)])
            else:
                indepent += 1
                if [ori1, des1] not in new_overlap_od:
                    new_overlap_od.append([ori1, des1])
                if [ori2, des2] not in new_overlap_od:
                    new_overlap_od.append([ori2, des2])
        if len(all_overlap_od) % 2 == 1:
            final_ori2 = all_overlap_od[-1][0]
            final_des2 = all_overlap_od[-1][1]
            final_ori1 = all_overlap_od[-2][0]
            final_des1 = all_overlap_od[-2][1]
            if final_ori2 > final_des1:
                if all_overlap_od[-1] not in new_overlap_od:
                    new_overlap_od.append(list(all_overlap_od[-1]))
            if final_ori2 <= final_des1 or distance(all_point[final_ori2], all_point[final_des1]) <= 5:
                if [min(final_ori1, final_ori2), max(final_des1, final_des2)] not in new_overlap_od:
                    new_overlap_od.append([min(final_ori1, final_ori2), max(final_des1, final_des2)])
        return new_overlap_od, indepent


def direction(overlap1, overlap2):
    slope1 = (overlap1[1][1] - overlap1[0][1]) / (overlap1[1][0] - overlap1[0][0])
    slope2 = (overlap2[1][1] - overlap2[0][1]) / (overlap2[1][0] - overlap2[0][0])
    simi_cos = (1 + slope1 * slope2) / math.sqrt(1 + slope1 ** 2) / math.sqrt(1 + slope2 ** 2)
    return simi_cos


def distance(a, b):
    return math.sqrt(math.pow((a[0] - b[0]), 2) + math.pow((a[1] - b[1]), 2))


def single_dump(all_utm_time, hmap, width, max_vel_str, max_vel_cur, gps, autovel, weight, is_local, id4node, id4seg):
    speed = 5
    offset4seg = id4node
    all_utm_time[:, -1] = np.array(range(len(all_utm_time)))
    all_utm_time_walk = all_utm_time
    all_utm_time_walk[:, -1] = all_utm_time_walk[:, -1] * speed
    # step0-1 calculate the vel of every point
    # With or Without vel
    if autovel == 0:
        new_all_utm_vel = all_utm_time
    else:
        all_point_for_vel = all_utm_time.T
        dif = np.diff(all_point_for_vel)
        all_vel = []
        for s in range(0, len(dif[0])):
            if dif[3][s] < 0.001:
                vel_of_point = 3
            else:
                vel_of_point = 3.6 * math.sqrt(
                    math.pow(dif[0][s], 2) + math.pow(dif[1][s], 2) + math.pow(dif[2][s], 2)) / dif[3][s]
            if vel_of_point < 3:
                vel_of_point = 3
            if vel_of_point > 40:
                vel_of_point = 40
            all_vel.append(vel_of_point)

        all_vel = list(filter(None, all_vel))
        all_vel = np.array(all_vel)
        N = 20
        weights = np.ones(N) / N
        s = np.convolve(weights, all_vel)[N - 1:-N + 1]

        all_utm_vel = list(all_utm_time)
        new_all_utm_vel = []
        for d in range(len(all_utm_vel)):
            new_d = list(all_utm_vel[d])
            if d >= len(s) - 1:
                new_d.append(np.max(s))
            elif d == 0:
                new_d.append(np.max(s))
            else:
                new_d.append(s[d])
            new_all_utm_vel.append(new_d)

        new_all_utm_vel = np.array(new_all_utm_vel)

    all_pure_utm = all_utm_time[:, :3]
    # step1 extract key point
    # step1-0 few points can be processed directly
    if len(new_all_utm_vel) <= 100:
        mask = rdp(all_pure_utm, epsilon=0.02, algo="iter", return_mask=True)
        all_point_final = new_all_utm_vel[mask]
    else:
        # step1-1 cluster by distance and time
        point_num = 100
        n_clusters = len(all_utm_time) // point_num
        kmeans = MiniBatchKMeans(n_clusters, init_size=n_clusters, random_state=1).fit(all_utm_time_walk)
        labels = kmeans.labels_

        # step2 sort all points by time order
        all_point = []
        for i in range(n_clusters):
            class_index = labels
            same_class_index = np.where(class_index == i)
            # all_utm_time_walk is for cluster and the useful one is new_all_utm_vel
            same_class_utm = all_pure_utm[list(same_class_index[0])]
            same_class = new_all_utm_vel[list(same_class_index[0])]
            mask4class = rdp(same_class_utm, epsilon=0.02, algo="iter", return_mask=True)
            keypoint4class = same_class[mask4class]
            if all_point == []:
                all_point = keypoint4class
            else:
                all_point = np.vstack((all_point, keypoint4class))
        arg = np.argsort(all_point[:, 3])
        all_point_final = all_point[list(arg)]
        # all_point_final = all_point

    # step3 delete the latter point in every short segment that length is less than 0.5m
    all_point_final = delete_near_point(all_point_final, 0.5)
    all_point_final = np.array(all_point_final)
    all_point_final = delete_near_point(all_point_final, 0.5)
    all_point_final = np.array(all_point_final)
    # Local cooridinate or GPS With or Without vel(2*2)
    vel_for_point = []
    lati = []
    longi = []
    height = []
    all_x = []
    all_y = []
    jw = []
    if gps == 0 and autovel == 0:
        for i in all_point_final:
            longi.append(i[0])
            lati.append(i[1])
            height.append(i[2])
            all_x.append(i[0])
            all_y.append(i[1])
            jw.append([i[0], i[1]])
    elif gps == 0 and autovel == 1:
        for i in all_point_final:
            longi.append(i[0])
            lati.append(i[1])
            height.append(i[2])
            all_x.append(i[0])
            all_y.append(i[1])
            jw.append([i[0], i[1]])
        vel_for_point = all_point_final[:, -1]
    elif gps == 1 and autovel == 0:
        lati = list(all_point_final[:, -2])
        longi = list(all_point_final[:, -3])
        all_x = all_point_final[:, 0].tolist()
        all_y = all_point_final[:, 1].tolist()
        height = list(all_point_final[:, 2])
        jw = all_point_final[:, -2:].tolist()
    elif gps == 1 and autovel == 1:
        lati = list(all_point_final[:, -2])
        longi = list(all_point_final[:, -3])
        all_x = all_point_final[:, 0].tolist()
        all_y = all_point_final[:, 1].tolist()
        height = list(all_point_final[:, 2])
        vel_for_point = all_point_final[:, -1]
        jw = all_point_final[:, -2:].tolist()
    # step4 Concat straight line
    # num = 0
    all_point_final = all_point_final.tolist()
    while True:
        slope = calculate_slope(all_point_final)
        after_concat = concat_stright(slope, all_point_final)
        #after_concat = list(after_concat)
        if after_concat == all_point_final or abs(len(after_concat) - len(all_point_final)) == 1:
            break
        else:
            all_point_final = after_concat
        # num += 1
    # Local Coordinate or GPS
    X = []
    Y = []
    key_jw = []
    if gps == 0 or is_local == 1:
        for i in all_point_final:
            X.append(i[0])
            Y.append(i[1])
            key_jw.append([i[0], i[1]])
    else:
        final_point = np.array(all_point_final)
        key_jw = final_point[:, -2:].tolist()
        X = list(final_point[:, 0])
        Y = list(final_point[:, 1])

    # step5 Judge the back off
    possible_backid = []
    possible_backvec = []
    for p in range(len(jw) - 2):
        vector1 = [all_x[p + 1] - all_x[p], all_y[p + 1] - all_y[p]]
        vector2 = [all_x[p + 2] - all_x[p + 1], all_y[p + 2] - all_y[p + 1]]
        back_cos = vector_cos(vector1, vector2)
        if back_cos < 0:
            possible_backid.append(p + 1)
            possible_backvec.append([vector1, vector2])
    # Check the back off id
    all_possible_backid = possible_backid[:]
    possible_backid = []
    for b in range(len(all_possible_backid) - 1):
        if vector_cos(possible_backvec[b][1], possible_backvec[b + 1][0]) > 0:
            possible_backid.append(all_possible_backid[b])
            possible_backid.append(all_possible_backid[b + 1])
        else:
            pass

    # step7-1 node_set
    # With or Without vel
    if autovel == 0:
        for i in range(len(lati)):
            node_lat = lati[i]
            node_lng = longi[i]
            node_hei = height[i]
            node = {"gps_weight": weight[0], "id": i + id4node, "lat": node_lat, "lng": node_lng,
                    "lslam_carto_weight": weight[3],
                    "name": str(i + id4node),
                    "qrcode_weight": weight[1], "radius": 0, "type": 17, "vslam_weight": weight[2],
                    "z": node_hei}
            hmap["node_set"].append(node)
        id4node += len(lati)
    else:
        for i in range(len(lati)):
            node_lat = lati[i]
            node_lng = longi[i]
            node_hei = height[i]
            node_vel = vel_for_point[i]
            node = {"gps_weight": weight[0], "id": i + id4node, "lat": node_lat, "lng": node_lng, "max_vel": node_vel,
                    "lslam_carto_weight": weight[3], "name": str(i + id4node),
                    "qrcode_weight": weight[1], "radius": 0, "type": 17, "vslam_weight": weight[2],
                    "z": node_hei}
            hmap["node_set"].append(node)
        id4node += len(lati)
    # step7-2 segment_set
    # Count the number of points between key points
    all_point_num = []
    for j in range(0, len(key_jw) - 1):
        point0 = key_jw[j]
        point1 = key_jw[j + 1]
        index0 = jw.index(point0)
        index1 = jw.index(point1)
        point_num = index1 - index0
        all_point_num.append(point_num)
    # Find the stright line which number of points is lager than 2 and distance is larger than 10m
    all_str_id = [k for k in range(len(all_point_num)) if all_point_num[k] > 1]
    real_str_id = []
    for h in range(len(all_str_id)):
        key_id0 = all_str_id[h]
        key_id1 = all_str_id[h] + 1
        if key_id1 > len(jw) - 1:
            break
        else:
            str_dis = distance([X[key_id0], Y[key_id0]], [X[key_id1], Y[key_id1]])
            if str_dis >= 5:
                real_str_id.append(all_str_id[h])

    all_segment_id = []
    # With or Without vel
    if autovel == 0:
        vel_str = max_vel_str
        vel_cur = max_vel_cur
    else:
        vel_str = 40
        vel_cur = 40
    # Judge whether this a pure curve
    if len(real_str_id) == 0:
        all_segment_id.append(key_jw[0])
        all_segment_id.append(key_jw[-1])
        only_cur_segment = {"id": id4seg, "lane_list": [
            {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": vel_cur,
             "name": "Path" + str(id4seg),
             "node_list": list(range(0 + offset4seg, len(jw) + offset4seg)), "right_line_type": 1, "seg_id": id4seg}],
                            "name": "seg" + str(id4seg)}
        id4seg += 1
        hmap["segment_set"].append(only_cur_segment)
    else:
        # Distinguish the stright and curve
        for p in range(len(real_str_id) - 1):
            first_id = jw.index(key_jw[real_str_id[p]])
            second_id = jw.index(key_jw[real_str_id[p] + 1])
            third_id = jw.index(key_jw[real_str_id[p + 1]])
            all_segment_id.append(first_id)
            if second_id == third_id:
                pass
            else:
                all_segment_id.append(second_id)
            # Generate the stright segment
            str_segment = {"id": id4seg, "lane_list": [
                {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": vel_str, "name": "Path" + str(id4seg),
                 "node_list": list(range(first_id + offset4seg, second_id + 1 + offset4seg)), "right_line_type": 1,
                 "seg_id": id4seg}],
                           "name": "seg" + str(id4seg)}
            id4seg += 1
            hmap["segment_set"].append(str_segment)
            # Generate the curve segment if second id is not the same as third id
            if second_id == third_id:
                pass
            else:
                cur_segment = {"id": id4seg, "lane_list": [
                    {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": vel_cur,
                     "name": "Path" + str(id4seg),
                     "node_list": list(range(second_id + offset4seg, third_id + 1 + offset4seg)), "right_line_type": 1,
                     "seg_id": id4seg}],
                               "name": "seg" + str(id4seg)}
                id4seg += 1
                hmap["segment_set"].append(cur_segment)

        # Append the first segment
        f_id = jw.index(key_jw[real_str_id[0]])
        if f_id == 0:
            pass
        else:
            # Generate the first curve segment
            first_cur_segment = {"id": id4seg, "lane_list": [
                {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": vel_cur, "name": "Path" + str(id4seg),
                 "node_list": list(range(0 + offset4seg, f_id + 1 + offset4seg)), "right_line_type": 1,
                 "seg_id": id4seg}],
                                 "name": "seg" + str(id4seg)}
            id4seg += 1
            hmap["segment_set"].insert(0, first_cur_segment)
            all_segment_id.insert(0, 0)
        # Append the last segment
        last_id = jw.index(key_jw[real_str_id[- 1]])
        all_segment_id.append(last_id)
        if last_id == len(jw) - 1:
            pass
        else:
            # Generate the last curve segment
            last_cur_segment = {"id": id4seg, "lane_list": [
                {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": vel_cur, "name": "Path" + str(id4seg),
                 "node_list": list(range(last_id + offset4seg, len(jw) + offset4seg)), "right_line_type": 1,
                 "seg_id": id4seg}],
                                "name": "seg" + str(id4seg)}
            id4seg += 1
            hmap["segment_set"].append(last_cur_segment)
            all_segment_id.append(len(jw) - 1)

    return hmap, id4node, id4seg
