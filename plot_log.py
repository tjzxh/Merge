import matplotlib.pyplot as plt
from process_log import process_pcmap
import process_log
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import datetime

gps = 0
vslam = 0
all_data0 = process_pcmap("44.pcmap", vslam, gps)
all_data1 = process_pcmap("55.pcmap", vslam, gps)
all_x0 = all_data0[:, 0]
all_y0 = all_data0[:, 1]
all_x1 = all_data1[:, 0]
all_y1 = all_data1[:, 1]
all_overlap, class_num = process_log.overlap_points(all_data0, all_data1)

id4node = 0
id4seg = 0
width = 3
max_vel_str = 20
max_vel_cur = 15
gps = 0
autovel = 0
weight = [1, 0, 0, 0]
is_local = 0
if gps == 0:
    is_coordinate_gps = 0
else:
    is_coordinate_gps = 1
hmap = {"debug_info": {"lat_bias": 0, "lng_bias": 0}, "is_coordinate_gps": is_coordinate_gps,
        "lane_switch_set": [],
        "node_set": [], "segment_set": []}
if all_overlap == [] or not all_overlap:
    ano_hmap = {"debug_info": {"lat_bias": 0, "lng_bias": 0}, "is_coordinate_gps": is_coordinate_gps,
                "lane_switch_set": [],
                "node_set": [], "segment_set": []}
    hmap, _, _ = process_log.single_dump(all_data0, hmap, width, max_vel_str, max_vel_cur, gps,
                                         autovel, weight, is_local, 0, 0)
    ano_hmap, _, _ = process_log.single_dump(all_data1, ano_hmap, width, max_vel_str, max_vel_cur, gps,
                                             autovel, weight, is_local, 0, 0)
    hmap = process_log.simple_merge(hmap, ano_hmap)
    print("There is no overlap, only simple merge!")
else:
    overlap_od0, overlap_od1 = process_log.overlap_od(all_overlap, class_num, all_data0, all_data1)
    real_ov_od0 = []
    real_ov_od1 = []
    # Generate a whole json file for hmap
    len0 = len(all_data0)
    id4connect0 = 0
    id4connect1 = 0

    # No overlap, then simple merge
    if not overlap_od0 or not overlap_od1:
        ano_hmap = {"debug_info": {"lat_bias": 0, "lng_bias": 0}, "is_coordinate_gps": is_coordinate_gps,
                    "lane_switch_set": [],
                    "node_set": [], "segment_set": []}
        hmap, _, _ = process_log.single_dump(all_data0, hmap, width, max_vel_str, max_vel_cur, gps,
                                             autovel, weight, is_local, 0, 0)
        ano_hmap, _, _ = process_log.single_dump(all_data1, ano_hmap, width, max_vel_str, max_vel_cur, gps,
                                                 autovel, weight, is_local, 0, 0)
        hmap = process_log.simple_merge(hmap, ano_hmap)
        print("There is no overlap, only simple merge!")
    else:
        # display overlap seg
        for p in range(len(overlap_od0)):
            fig = plt.figure()
            overlap_data0 = all_data0[overlap_od0[p][0]:overlap_od0[p][1]]
            overlap_data1 = all_data1[overlap_od1[p][0]:overlap_od1[p][1]]
            plt.subplot(121)
            plt.plot(all_x0, all_y0, c='b')
            plt.scatter(overlap_data0[:, 0], overlap_data0[:, 1], c='r')
            plt.subplot(122)
            plt.plot(all_x1, all_y1, c='b')
            plt.scatter(overlap_data1[:, 0], overlap_data1[:, 1], c='lime')
            plt.show()
        # handle all the overlap seg
        # if the original point of overlap is the first point of trajectory
        for d in range(len(overlap_od0)):
            # the start of overlap
            if d == 0:
                # find the real start of overlap
                if process_log.distance(all_data0[overlap_od0[0][0]][:2], all_data0[0][:2]) < 0.5:
                    overlap_od0[0][0] = 0
                if process_log.distance(all_data1[overlap_od1[0][0]][:2], all_data1[0][:2]) < 0.5:
                    overlap_od1[0][0] = 0
                if overlap_od0[0][0] == 0 and overlap_od1[0][0] == 0:
                    # the first seg is the overlap seg
                    ov_st = 0
                    ov_st1 = 0
                else:
                    ov_st = overlap_od0[0][0]
                    ov_st1 = overlap_od1[0][0]
                    if overlap_od0[0][0] != 0:
                        vector0 = [all_data0[ov_st][0] - all_data0[ov_st - 1][0],
                                   all_data0[ov_st][1] - all_data0[ov_st - 1][1]]
                    if overlap_od1[0][0] != 0:
                        vector1 = [all_data0[ov_st][0] - all_data1[ov_st1 - 1][0],
                                   all_data0[ov_st][1] - all_data1[ov_st1 - 1][1]]
                    vector4connect = [all_data0[ov_st + 1][0] - all_data0[ov_st][0],
                                      all_data0[ov_st + 1][1] - all_data0[ov_st][1]]
                    if overlap_od0[0][0] != 0:
                        cos0 = cosine_similarity([vector0], [vector4connect])
                        id0 = 0
                        while cos0[0] < 0.8:
                            id0 += 1
                            ov_st += 1
                            vector4connect = [all_data0[ov_st + 1][0] - all_data0[ov_st][0],
                                              all_data0[ov_st + 1][1] - all_data0[ov_st][1]]
                            cos0 = cosine_similarity([vector0], [vector4connect])
                            if id0 > 10:
                                break
                    if overlap_od1[0][0] != 0:
                        id1 = 0
                        cos1 = cosine_similarity([vector1], [vector4connect])
                        min_dis = process_log.distance(all_data0[ov_st][:2], all_data1[ov_st1 - 1][:2])
                        while cos1[0] < 0.8:
                            if id1 > 10 or min_dis < 0.1:
                                break
                            ov_st += 1
                            id1 += 1
                            cos1_cp = cos1
                            vector4connect = [all_data0[ov_st + 1][0] - all_data0[ov_st][0],
                                              all_data0[ov_st + 1][1] - all_data0[ov_st][1]]
                            cos1 = cosine_similarity([vector1], [vector4connect])
                            # _, ov_st1 = process_log.find_neighbor(all_data1, np.array(
                            #     [all_data0[ov_st][0], all_data0[ov_st][1]]).reshape(1, 2))
                            # ov_st1 = ov_st1[0]
                            min_dis = process_log.distance(all_data0[ov_st][:2], all_data1[ov_st1 + 1][:2])
                    # dump the first seg
                    if overlap_od0[0][0] != 0:
                        first_data0 = all_data0[:ov_st]
                        hmap, id4node, id4seg = process_log.single_dump(first_data0, hmap, width, max_vel_str, max_vel_cur,
                                                                        gps,
                                                                        autovel,
                                                                        weight, is_local, id4node, id4seg)
                        id4connect0 = id4node - 1

                    if overlap_od1[0][0] != 0:
                        first_data1 = all_data1[:ov_st1]
                        hmap, id4node, id4seg = process_log.single_dump(first_data1, hmap, width, max_vel_str, max_vel_cur,
                                                                        gps,
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

                # find the real overlap end
                ov_ed, ov_ed1 = process_log.find_ed_in_overlap(overlap_od0[d][1], all_data0, all_data1)
                # dump the overlap seg
                real_ov_od0.append([ov_st, ov_ed])
                real_ov_od1.append([ov_st1, ov_ed1])
                overlap0 = all_data0[ov_st:ov_ed]
                hmap, id4node, id4seg = process_log.single_dump(overlap0, hmap, width, max_vel_str, max_vel_cur, gps,
                                                                autovel, weight, is_local, id4node, id4seg)

            # the middle part of overlap
            if len(overlap_od0) > 1 and d != 0:
                id4connect_f = id4node - 1
                ov_st, ov_st1 = process_log.find_st_in_overlap(overlap_od0[d][0], all_data0, all_data1)
                ov_ed, ov_ed1 = process_log.find_ed_in_overlap(overlap_od0[d][1], all_data0, all_data1)
                real_ov_od0.append([ov_st, ov_ed])
                real_ov_od1.append([ov_st1, ov_ed1])
                data0 = all_data0[real_ov_od0[d - 1][1]:real_ov_od0[d][0]]
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
                data1 = all_data1[real_ov_od1[d - 1][1]:real_ov_od1[d][0]]
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
                overlap = all_data0[real_ov_od0[d][0]:real_ov_od0[d][1]]
                hmap, id4node, id4seg = process_log.single_dump(overlap, hmap, width, max_vel_str, max_vel_cur, gps,
                                                                autovel, weight, is_local, id4node, id4seg)
            # the last overlap
            if d == len(overlap_od0) - 1:
                id4final = id4node - 1
                ov_st, ov_st1 = process_log.find_st_in_overlap(overlap_od0[d][0], all_data0, all_data1)
                ov_ed, ov_ed1 = process_log.find_ed_in_overlap(overlap_od0[d][1], all_data0, all_data1)
                real_ov_od0.append([ov_st, ov_ed])
                real_ov_od1.append([ov_st1, ov_ed1])
                if process_log.distance(all_data0[ov_ed][:2], all_data0[-1][:2]) < 0.5:
                    ov_ed = -1
                if process_log.distance(all_data1[ov_ed1][:2], all_data1[-1][:2]) < 0.5:
                    ov_ed1 = -1
                if ov_ed != -1:
                    final_connect_seg0 = {"id": id4seg, "lane_list": [
                        {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
                         "name": "Path" + str(id4seg),
                         "node_list": [id4final, id4node], "right_line_type": 1,
                         "seg_id": id4seg}], "name": "seg" + str(id4seg)}
                    id4seg += 1
                    hmap["segment_set"].append(final_connect_seg0)
                    final_data0 = all_data0[ov_ed:]
                    hmap, id4node, id4seg = process_log.single_dump(final_data0, hmap, width, max_vel_str,
                                                                    max_vel_cur, gps,
                                                                    autovel, weight, is_local, id4node, id4seg)
                if ov_ed1 != -1:
                    final_connect_seg1 = {"id": id4seg, "lane_list": [
                        {"id": 0, "lane_width": width, "left_line_type": 1, "max_vel": max_vel_str,
                         "name": "Path" + str(id4seg),
                         "node_list": [id4final, id4node], "right_line_type": 1,
                         "seg_id": id4seg}], "name": "seg" + str(id4seg)}
                    id4seg += 1
                    hmap["segment_set"].append(final_connect_seg1)
                    final_data1 = all_data1[ov_ed1:]
                    hmap, id4node, id4seg = process_log.single_dump(final_data1, hmap, width, max_vel_str,
                                                                    max_vel_cur, gps,
                                                                    autovel, weight, is_local, id4node, id4seg)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(MyEncoder, self).default(obj)


with open(datetime.datetime.now().strftime('%H-%M-%S') + '.hmap', 'w') as f1:
    json.dump(hmap, f1, indent=4, cls=MyEncoder)
