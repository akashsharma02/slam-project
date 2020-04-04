# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/ReconstructionSystem/make_fragments.py

import numpy as np
import math
import copy
import open3d as o3d
import sys

from matplotlib import pyplot as plt
from skimage import morphology
from optimize_posegraph import optimize_posegraph_for_fragment
from utility.file import join, make_clean_folder, get_rgbd_file_lists, get_mask_file_lists
from utility.opencv import initialize_opencv
sys.path.append(".")

# check opencv python package
with_opencv = initialize_opencv()
if with_opencv:
    from opencv_pose_estimation import pose_estimation


def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_trunc=config["max_depth"],
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image


class ObjectLandmark(object):
    def __init__(self, rgbd, pcd, intrinsic, label, score, obj_mask):
        self.rgbd = rgbd
        self.pcd = pcd
        self.intrinsic = intrinsic
        self.label = label
        self.score = score
        self.mask = obj_mask

    def check_registration(self, other_obj, pose, config):
        # Failure if they are not the same label
        if self.label != other_obj.object_landmark.label:
            return False

        other_pcd = copy.deepcopy(other_obj.object_landmark.pcd)
        other_pcd.transform(np.linalg.inv(pose))

        other_pcd_points = np.asarray(other_pcd.points) / np.asarray(
            other_pcd.points)[:, 2][:, None]
        other_pcd_points2d = np.asarray(np.round(
            (self.intrinsic.intrinsic_matrix @ other_pcd_points.T).T),
                                        dtype=int)[:, 0:2]
        other_pcd_points2d = other_pcd_points2d[
            (other_pcd_points2d[:, 0] > 20)
            & (other_pcd_points2d[:, 0] < self.intrinsic.width - 20)]
        other_pcd_points2d = other_pcd_points2d[
            (other_pcd_points2d[:, 1] > 20)
            & (other_pcd_points2d[:, 1] < self.intrinsic.height - 20)]
        proj_mask = np.zeros_like(other_obj.object_landmark.mask)
        proj_mask[other_pcd_points2d[:, 1], other_pcd_points2d[:, 0]] = True
        proj_mask = morphology.binary_dilation(proj_mask)

        confidence = np.sum(proj_mask & self.mask) / np.sum(self.mask)

        # print(confidence)
        if confidence < config["obj_confidence_threshold"]:
            return False
        return True


class ReferenceObject(object):
    def __init__(self, object_landmark, object_prior_pose, config):
        self.object_landmark = object_landmark
        self.object_prior_pose = object_prior_pose
        self.volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=config["object_tsdf_cubic_size"] / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8)

        self.existence_count = 0
        self.no_existence_count = 1
        self.rgbd_list = []
        self.pose_list = []

    def add(self, obj, frame_pose):
        self.rgbd_list.append(obj.rgbd)
        self.pose_list.append(frame_pose)
        assert len(self.rgbd_list) == len(self.pose_list)

    def integrate(self):
        self.volume.integrate(self.object_landmark.rgbd,
                              self.object_landmark.intrinsic,
                              self.object_prior_pose)
        for i in range(len(self.rgbd_list)):
            # print(
            #     f"integrate object: {self.object_landmark.label} {i} ({i} of {len(self.rgbd_list)})"
            # )
            self.volume.integrate(self.rgbd_list[i],
                                  self.object_landmark.intrinsic,
                                  self.pose_list[i])

    def add_observation(self):
        self.existence_count += 1

    def remove_observation(self):
        self.no_existence_count += 1

    def existence_expectation(self):
        return self.existence_count / (self.existence_count +
                                       self.no_existence_count)


def read_mask_objects(color_file, depth_file, mask_file, label_file, intrinsic,
                      config):
    mask = o3d.io.read_image(mask_file)
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    labels = np.loadtxt(label_file, delimiter=',', ndmin=2)
    # print(labels, labels.shape)
    # If there are no objects in the frame
    if labels.shape[1] == 0 or labels.shape[0] == 0:
        return None

    label = labels[:, 0]
    scores = labels[:, 1]
    labels = list(map(int, label))
    scores = list(map(float, scores))

    mask_buf = np.asarray(mask)
    objects = []
    for i, label in enumerate(labels):
        select_mask = np.full_like(mask_buf, 2**i, dtype=np.uint16)
        object_mask = np.bitwise_and(mask_buf, select_mask)
        background_i = np.where(object_mask >= 1, False, True)
        depth_i = copy.deepcopy(depth)
        np.asarray(depth_i)[background_i] = 0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_i, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        objects.append(
            ObjectLandmark(rgbd, pcd, intrinsic, label, scores[i],
                           ~background_i))

    return objects


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, config):
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], True,
                                        config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], True,
                                        config)

    option = o3d.odometry.OdometryOption()
    option.max_depth_diff = config["max_depth_diff"]
    if abs(s - t) is not 1:
        if with_opencv:
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:
                [success, trans, info] = o3d.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(s - sid,
                                                   t - sid,
                                                   trans,
                                                   info,
                                                   uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(s - sid,
                                                       t - sid,
                                                       trans,
                                                       info,
                                                       uncertain=True))
    o3d.io.write_pose_graph(
        join(path_dataset,
             config["template_fragment_posegraph"] % fragment_id), pose_graph)


def integrate_mask_frames_for_fragment(color_files, depth_files, mask_files,
                                       label_files, fragment_id, n_fragments,
                                       pose_graph_name, intrinsic, config):

    pose_graph = o3d.io.read_pose_graph(pose_graph_name)

    reference_obj_pool = {}
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        curr_objects = read_mask_objects(color_files[i_abs],
                                         depth_files[i_abs], mask_files[i_abs],
                                         label_files[i_abs], intrinsic, config)
        frame_pose = pose_graph.nodes[i].pose
        # If there are no objects in the frame
        if curr_objects == None:
            continue

        for curr_obj in curr_objects:
            reference_found = False
            if reference_obj_pool:
                curr_obj_registered = False
                # print(f"Current object label: {curr_obj.label}")
                for obj_id, reference_obj in reference_obj_pool.items():
                    success = curr_obj.check_registration(
                        reference_obj, frame_pose, config)
                    if success:
                        if not curr_obj_registered:
                            curr_obj_registered = True
                            reference_found = True
                            reference_obj.add_observation()
                            # print(
                            #     f"Adding observation of reference obj:{reference_obj.object_landmark.label} with current object:{curr_obj.label} at frame{i}"
                            # )
                            reference_obj.add(curr_obj,
                                              np.linalg.inv(frame_pose))
                        else:
                            reference_obj.remove_observation()

            if not reference_found:
                if not reference_obj_pool:
                    obj_id = 0
                else:
                    obj_id = max(reference_obj_pool.keys()) + 1

                reference_obj = ReferenceObject(curr_obj,
                                                np.linalg.inv(frame_pose),
                                                config)
                reference_obj_pool[obj_id] = reference_obj
                # print(
                #     f"Reference object added: {obj_id}: {reference_obj.object_landmark.label} in frame: {i}"
                # )

    reference_objs_to_remove = []
    for obj_id, reference_obj in reference_obj_pool.items():
        # print(
        #     f" For {obj_id}:{reference_obj.object_landmark.label} Existence expectation of the object: {reference_obj.existence_expectation()}"
        # )

        if reference_obj.existence_expectation() < 0.25:
            reference_objs_to_remove.append(obj_id)

    for obj_id in reference_objs_to_remove:
        reference_obj_pool.pop(obj_id)
    # print(f"Size of the reference pool: {len(reference_obj_pool.keys())}")
    obj_meshes = {}
    for obj_id, reference_obj in reference_obj_pool.items():
        reference_obj.integrate()

        mesh = reference_obj.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        obj_meshes[reference_obj.object_landmark.label] = mesh

    return obj_meshes


def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    # print(f"Length of pose graph nodes: {len(pose_graph.nodes)}")
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
              (fragment_id, n_fragments - 1, i_abs, i + 1, len(
                  pose_graph.nodes)))
        rgbd = read_rgbd_image(color_files[i_abs], depth_files[i_abs], False,
                               config)
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 mask_files, label_files, fragment_id,
                                 n_fragments, intrinsic, config):

    obj_meshes = integrate_mask_frames_for_fragment(
        color_files, depth_files, mask_files, label_files, fragment_id,
        n_fragments,
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)

    make_clean_folder(
        join(config["path_dataset"], config["folder_fragment"],
             f"fragment_{fragment_id:03d}"))
    for idx, (label, mesh) in enumerate(obj_meshes.items()):
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd_name = config["template_fragment_object_pointcloud"]
        pcd_name = join(path_dataset, pcd_name % (fragment_id, label, idx))
        # print(f"Writing object pcd to file: {pcd_name}")
        o3d.io.write_point_cloud(pcd_name, pcd, False, True)

    mesh = integrate_rgb_frames_for_fragment(
        color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,
             config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = join(path_dataset,
                    config["template_fragment_pointcloud"] % fragment_id)
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)


def process_single_fragment(fragment_id, color_files, depth_files, mask_files,
                            label_files, n_files, n_fragments, config):
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config)

    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id,
                                    config)

    make_pointcloud_for_fragment(config["path_dataset"], color_files,
                                 depth_files, mask_files, label_files,
                                 fragment_id, n_fragments, intrinsic, config)


def run(config):
    print("making fragments from RGBD sequence.")
    make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))
    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    [mask_files, label_files] = get_mask_file_lists(config["path_dataset"])
    n_files = len(color_files)
    n_mask_files = len(mask_files)
    print(n_files, n_mask_files)
    assert n_files == n_mask_files == n_mask_files

    n_fragments = int(
        math.ceil(float(n_files) / config['n_frames_per_fragment']))

    if config["python_multi_threading"]:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
        Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(
            fragment_id, color_files, depth_files, mask_files, label_files,
            n_files, n_fragments, config)
                                    for fragment_id in range(n_fragments))
    else:
        for fragment_id in range(n_fragments):
            process_single_fragment(fragment_id, color_files, depth_files,
                                    n_files, n_fragments, config)
