# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/ReconstructionSystem/register_fragments.py

import numpy as np
import open3d as o3d
import sys
import os
import copy
from utility.file import join, get_file_list, make_clean_folder
from utility.visualization import draw_registration_result
sys.path.append(".")
from optimize_posegraph import optimize_posegraph_for_scene
from refine_registration import multiscale_icp


def preprocess_point_cloud(pcd, config):
    voxel_size = config["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)


def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh,
                              config):
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "fgr":
        result = o3d.registration.registration_fast_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            o3d.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    if config["global_registration"] == "ransac":
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    if (result.transformation.trace() == 4.0):
        return (False, 0, np.identity(4), np.zeros((6, 6)))
    information = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result.transformation)
    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return (False, 0, np.identity(4), np.zeros((6, 6)))
    return (True, result.fitness, result.transformation, information)


def compute_initial_registration_odom(s, t, source_down, target_down, source_fpfh,
                                 target_fpfh, path_dataset, config):

    print("Using RGBD odometry")
    pose_graph_frag = o3d.io.read_pose_graph(
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % s))
    n_nodes = len(pose_graph_frag.nodes)
    transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes -
                                                              1].pose)
    (fitness, transformation, information) = \
            multiscale_icp(source_down, target_down,
            [config["voxel_size"]], [50], config, transformation_init)

    if config["debug_mode"]:
        draw_registration_result(source_down, target_down, transformation)

    return (transformation, information)

def compute_initial_registration_loop(source_down, target_down, source_fpfh, target_fpfh, config):
    (success, fitness, transformation,
     information) = register_point_cloud_fpfh(source_down, target_down,
                                              source_fpfh, target_fpfh,
                                              config)
    if not success:
        print("No reasonable solution. Skip this pair")
        return (False, 0, np.identity(4), np.zeros((6, 6)))

    if config["debug_mode"]:
        draw_registration_result(source_down, target_down, transformation)

    return (True, fitness, transformation, information)


def update_posegraph_for_scene(s, t, transformation, information, assocs, odometry,
                               pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(s,
                                           t,
                                           transformation,
                                           information,
                                           uncertain=False))
    else:  # loop closure case
        for i in range(len(assocs)):
            for j in range(len(assocs[i])):
                if assocs[i][j].weight > 0:
                    transformation = assocs[i][j].transformation
                    information = assocs[i][j].information
                    pose_graph.edges.append(
                            o3d.registration.PoseGraphEdge(s, t, transformation, information, uncertain=True))
                    print(assocs[i][j].transformation)
    return (odometry, pose_graph)


def read_obj_files(obj_file_names, s, t, config):
    print(obj_file_names[s], obj_file_names[t])
    source_obj, target_obj = [], []
    for source_obj_file in obj_file_names[s]:
        source_file_name = os.path.basename(source_obj_file)
        obj_class = os.path.splitext(source_file_name)[0].split('_')[1]
        current_source_obj = o3d.io.read_point_cloud(source_obj_file)
        if np.asarray(current_source_obj.points
                      ).shape[0] >= config["object_size_threshold"]:
            source_obj.append((int(obj_class), current_source_obj))

    for target_obj_file in obj_file_names[t]:
        target_file_name = os.path.basename(target_obj_file)
        obj_class = os.path.splitext(target_file_name)[0].split('_')[1]
        current_target_obj = o3d.io.read_point_cloud(target_obj_file)
        if np.asarray(current_target_obj.points
                      ).shape[0] >= config["object_size_threshold"]:
            target_obj.append((int(obj_class), current_target_obj))

    return source_obj, target_obj


def draw_pcd_registration(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([source_temp, target_temp, target_frame])


class Association:
    def __init__(self, label, weight, transformation, information):
        self.label = label
        self.weight = weight
        self.transformation = transformation
        self.information = information


def associate_obj_pairs(source_objs, target_objs, s, t, config):
    assocs = [[None for i in range(len(target_objs))]
              for j in range(len(source_objs))]
    for i, (s_label, source_obj) in enumerate(source_objs):
        for j, (t_label, target_obj) in enumerate(target_objs):
            if s_label != t_label:
                assocs[i][j] = Association(s_label, 0, np.identity(4),
                                           np.identity(6))
                continue

            (source_down,
             source_fpfh) = preprocess_point_cloud(source_obj, config)
            (target_down,
             target_fpfh) = preprocess_point_cloud(target_obj, config)
            (success, weight, transformation,
             information) = compute_initial_registration_loop(
                 source_down, target_down, source_fpfh, target_fpfh, config)

            if config["debug_mode"]:
                print(i, j, weight, transformation, information)
                draw_pcd_registration(source_obj, target_obj, transformation)
            assocs[i][j] = Association(s_label, weight, transformation,
                                       information)

    return np.asarray(assocs)


def register_point_cloud_pair(ply_file_names, obj_file_names, s, t, config):
    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])
    (source_down, source_fpfh) = preprocess_point_cloud(source, config)
    (target_down, target_fpfh) = preprocess_point_cloud(target, config)

    if t == s + 1:  # odometry cases
        (transformation, information) = \
                compute_initial_registration_odom(
                s, t, source_down, target_down,
                source_fpfh, target_fpfh, config["path_dataset"], config)

        if config["debug_mode"]:
            print(transformation)
            print(information)

        return (True, transformation, information, None)
    else:  #loop closure cases
        source_objs, target_objs = read_obj_files(obj_file_names, s, t, config)
        assocs = associate_obj_pairs(source_objs, target_objs, s, t, config)

        if config["debug_mode"]:
            for i in range(len(assocs)):
                for j in range(len(assocs[i])):
                    print(f"Label: {assocs[i][j].label}, weight: {assocs[i][j].label}")
                    print(f"Transform:\n {assocs[i][j].transformation}")

        return (True, np.identity(4), np.identity(6), assocs)

# other types instead of class?
class edge:
    def __init__(self, s, t):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = np.identity(4)
        self.infomation = np.identity(6)
        self.assocs = None


def make_posegraph_for_scene(ply_file_names, obj_file_names, config):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))

    n_files = len(ply_file_names)
    edges = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            edges[s * n_files + t] = edge(s, t)

    if config["python_multi_threading"]:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(), max(len(edges), 1))
        results = Parallel(n_jobs=MAX_THREAD)(
            delayed(register_point_cloud_pair)(
                ply_file_names, obj_file_names, edges[r].s, edges[r].t, config) for r in edges)
        for i, r in enumerate(edges):
            edges[r].success = results[i][0]
            edges[r].transformation = results[i][1]
            edges[r].information = results[i][2]
            edges[r].assocs = results[i][3]
    else:
        for r in edges:
            (edges[r].success, edges[r].transformation,
                    edges[r].information, edges[r].assocs) = \
                    register_point_cloud_pair(ply_file_names, obj_file_names,
                    edges[r].s, edges[r].t, config)
    for r in edges:
        if edges[r].success:
            (odometry, pose_graph) = update_posegraph_for_scene(
                edges[r].s, edges[r].t, edges[r].transformation,
                edges[r].information, edges[r].assocs, odometry, pose_graph)
    o3d.io.write_pose_graph(
        join(config["path_dataset"], config["template_global_posegraph"]),
        pose_graph)


def run(config):
    print("register fragments.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), "ply")

    n_files = len(ply_file_names)
    obj_file_names = []
    for i in range(n_files):
        obj_file_names.append(
            get_file_list(
                join(config["path_dataset"], config["folder_fragment"],
                     f"fragment_{i:03d}/"), "ply"))
    make_clean_folder(join(config["path_dataset"], config["folder_scene"]))
    make_posegraph_for_scene(ply_file_names, obj_file_names, config)
    optimize_posegraph_for_scene(config["path_dataset"], config)
