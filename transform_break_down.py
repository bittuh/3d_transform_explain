import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import open3d.visualization as vis
import math
from numpy import linalg as LA
import time
import copy
from itertools import chain

def get_axis(size=0.3):
    points = [
                [0, 0, 0],
                [size, 0, 0],
                [0, size, 0],
                [0, 0, size],
            ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
    ]
    colors = [[1, 0, 0], [0,1,0], [0,0,1] ] # RGB
    xyz_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    xyz_axis.colors = o3d.utility.Vector3dVector(colors)
    return xyz_axis

def create_vehicle(axis_size=1.0):
    body_frame = get_axis(size=axis_size)
    #obj = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.15,create_uv_map=True)
    #obj.translate(np.array([0,0,0.1]))

    dataset = o3d.data.BunnyMesh()
    obj = o3d.io.read_triangle_mesh(dataset.path)
    obj.scale(0.7, [0,0,0])
    obj.compute_vertex_normals()

    return obj, body_frame

def to_homo(*, rot=None, trans=None):
    homo = np.eye(4)
    if rot is not None:
        homo[:3,:3] = rot
    if trans is not None:
        homo[:3,3] = trans
    return homo

def get_homo_rot(rot_matrix):
    homo = np.eye(4)
    homo[:3, :3] = rot_matrix
    return homo

def get_rotation_matrix(roll, pitch, yaw, order='zyx'):
    T = Rotation.from_euler(order, [roll,pitch,yaw], degrees=True).as_matrix()
    return get_homo_rot(T), T

def get_ele_rotation_matrix(axis, degree):
    if axis == 'x':
        T = Rotation.from_euler('zyx', [0,0,degree], degrees=True).as_matrix()
    elif axis == 'y':
        T = Rotation.from_euler('zyx', [0,degree,0], degrees=True).as_matrix()
    elif axis == 'z':
        T = Rotation.from_euler('zyx', [degree,0,0], degrees=True).as_matrix()
    else:
        raise RuntimeError('invalid axis')

    return get_homo_rot(T), T

def get_homo_translate_matrix(x,y,z):
    T = np.eye(4)
    T[0,3] = x   
    T[1,3] = y
    T[2,3] = z
    return T

def get_homo_quaternion_matrix(q):
    print('jue quat:',q)
    T = Rotation.from_quat(q).as_matrix()
    print('jue mat:',T)
    return get_homo_rot(T), T

def get_pose_mat(quat, position):
    T = get_homo_rot(Rotation.from_quat(quat).as_matrix())
    T[0,3] = position[0]
    T[1,3] = position[1]
    T[2,3] = position[2]
    return T

def inter_keyframe(pose1, pose2):
    '''
    get pose P which pose1 -> P -> pose2 are rot and trans individually
    '''
    pose1_pose2_tf = np.dot(pose2, np.linalg.inv(pose1))
    _1to2_quat = Rotation.from_matrix(pose1_pose2_tf[:3,:3]).as_quat()

    another_1to2_quat = Rotation.from_matrix(np.dot(pose2[:3,:3],np.linalg.inv(pose1[:3,:3]))).as_quat()

    rot_only = copy.deepcopy(pose1_pose2_tf)
    rot_only[0,3] = 0
    rot_only[1,3] = 0
    rot_only[2,3] = 0
    keyframe = np.dot(rot_only, pose1) 
    return keyframe

def rot_animation(pose1, pose2, interval=30):

    # check only rotation happen
    diff_tf = np.dot(pose2, np.linalg.inv(pose1))
    assert(abs(diff_tf[0,3]) < 1e-15)
    assert(abs(diff_tf[1,3]) < 1e-15)
    assert(abs(diff_tf[2,3]) < 1e-15)

    diff_rotvec = Rotation.from_matrix(diff_tf[:3,:3]).as_rotvec()
    rot_step = Rotation.from_rotvec(diff_rotvec/float(interval))

    inter_pose = pose1
    for i in range(interval):
        inter_pose = np.dot(get_homo_rot(rot_step.as_matrix()), inter_pose)
        yield inter_pose, None
    #print('pose the same?', pose2, inter_pose)

def trans_animation(pose1, pose2, interval=30):

    # TODO
    # check only translation happen

    diff_tf = np.dot(pose2, np.linalg.inv(pose1))
    trans_step = np.array(diff_tf[0:3,3])/float(interval) # copy
    tf_step = np.eye(4)
    tf_step[0:3,3] = trans_step

    inter_pose = pose1
    for i in range(interval):
        inter_pose = np.dot(tf_step, inter_pose)

        yield inter_pose, None

def never_closed_visualizer(geometry_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geometry_list:
        vis.add_geometry(g)

    while True:
        vis.poll_events()
        vis.update_renderer()


def animation_display(static_geoms, animations, leaveTrace=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([.3,.3,.3])
    opt.line_width = 3.0 

    all_iter = chain(*animations)
    moving_axis = get_axis(size=0.15)
    #moving_obj, moving_axis = create_vehicle(axis_size=0.15) 

    for g in static_geoms:
        vis.add_geometry(g)
    vis.add_geometry(moving_axis)

    pose_save = []
    cache_added_obj = []

    while True:

        try:
            anim_pose, _ = next(all_iter)
            pose_save.append((anim_pose, None))
            moving_axis.transform(anim_pose)
            vis.update_geometry(moving_axis)
            vis.poll_events()
            vis.update_renderer()

            # leave trace
            if leaveTrace:
                anim_trace = copy.deepcopy(moving_axis)
                cache_added_obj.append(anim_trace)
                vis.add_geometry(anim_trace, reset_bounding_box=False)
            moving_axis.transform(np.linalg.inv(anim_pose))
            time.sleep(0.01)
        except StopIteration:
            for i in range(100):
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)
            all_iter = iter(pose_save) 
            pose_save = []

            if leaveTrace:
                for a in cache_added_obj:
                    vis.remove_geometry(a, reset_bounding_box=False)
                cache_added_obj = []
            


def main3():

    # scene setup
    orig_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    quat1 = Rotation.from_euler('zyx', [45,45,45]).as_quat()
    quat2 = Rotation.from_euler('zyx', [10, 45, 30]).as_quat()
    quat3 = Rotation.from_euler('zyx', [20,45,15]).as_quat()

    orig_pose = get_pose_mat([0,0,0,1], [0,0,0])
    key_pose0 = get_pose_mat([0,0,0,1], [1, 0.5, 0.5])
    key_pose1 = get_pose_mat(quat1, [2, 1, 1])
    key_pose2 = get_pose_mat(quat2, [4, 2, 2])

    straight_route = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([[0,0,0], [6,3,3]]),
            lines=o3d.utility.Vector2iVector([[0,1]]),
            )

    pose0_axis = get_axis().transform(key_pose0)
    pose1_axis = get_axis().transform(key_pose1)
    pose2_axis = get_axis().transform(key_pose2)

    # animation 1
    rot_anim_keypose1 = rot_animation(key_pose0, inter_keyframe(key_pose0, key_pose1))
    trans_anim_1 = trans_animation(inter_keyframe(key_pose0, key_pose1), key_pose1)

    # animation 2
    rot_anim_keypose2 = rot_animation(key_pose1, inter_keyframe(key_pose1, key_pose2))
    trans_anim_2 = trans_animation(inter_keyframe(key_pose1, key_pose2), key_pose2)

    animation_display([orig_frame, straight_route, pose0_axis, pose1_axis, pose2_axis], [rot_anim_keypose1, trans_anim_1, rot_anim_keypose2, trans_anim_2])


def break_down_rotation():

    orig_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    def transform_all(geoms, tf):
        return [g.transform(tf) for g in geoms]

    # init obj pose
    init_translate = to_homo(trans=np.array([0.3,0.15,-0.1]))  
    rot, rot_3x3 = get_rotation_matrix(90, 60, 60) 
    #rot_part1, _ = get_rotation_matrix(0,0,60)
    #rot_part2, _ = get_rotation_matrix(0,60,0)
    #rot_part3, _ = get_rotation_matrix(90,0,0)
    rot_part1, _ = get_rotation_matrix(0,0,90, order='xyz')
    rot_part2, _ = get_rotation_matrix(0,-60,0, order='xyz')
    rot_part3, _ = get_rotation_matrix(60,0,0, order='xyz')

    old_obj, old_obj_axis = create_vehicle(axis_size=0.15)
    pose1_obj, pose1_axis = copy.deepcopy(old_obj), copy.deepcopy(old_obj_axis)
    pose2_obj, pose2_axis = copy.deepcopy(old_obj), copy.deepcopy(old_obj_axis)
    #pose3_obj, pose3_axis = copy.deepcopy(old_obj), copy.deepcopy(old_obj_axis)
    old_obj.transform(init_translate)
    old_obj_axis.transform(init_translate)

    new_obj, new_obj_axis = create_vehicle(axis_size=0.15)
    new_obj.transform(init_translate).transform(rot)
    new_obj_axis.transform(init_translate).transform(rot)


    # keyframe
    inter_pose1 = np.dot(rot_part3, init_translate)
    inter_pose2 = np.dot(rot_part2, inter_pose1)
    inter_pose3 = np.dot(rot_part1, inter_pose2)

    # static obj for keyframe
    pose1_obj.transform(inter_pose1)
    pose1_axis.transform(inter_pose1)
    pose2_obj.transform(inter_pose2)
    pose2_axis.transform(inter_pose2)
    #pose3_obj.transform(inter_pose3)
    #pose3_axis.transform(inter_pose3)

    # animation
    anim_keypose1 = rot_animation(init_translate, inter_pose1, interval=250)
    anim_keypose2 = rot_animation(inter_pose1, inter_pose2, interval=250)
    anim_keypose3 = rot_animation(inter_pose2, inter_pose3, interval=250)
    all_anim = chain(anim_keypose1, anim_keypose2, anim_keypose3)

    animation_display([orig_frame, 
        old_obj, 
        old_obj_axis, 
        new_obj, 
        new_obj_axis,
        pose1_obj,
        pose1_axis,
        pose2_obj,
        pose2_axis], [anim_keypose1, anim_keypose2, anim_keypose3])

def demo_transform_from_origin():

    orig_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    orig_pose = get_pose_mat([0,0,0,1], [0,0,0])

    quat1 = Rotation.from_euler('zyx', [45,45,45]).as_quat()
    key_pose1 = get_pose_mat(quat1, [.2, 0.2, 0.1])

    inter_pose = inter_keyframe(orig_pose, key_pose1)


    # orig -> inter_pose -> key_pose1
    old_obj, old_obj_axis = create_vehicle(axis_size=0.15)
    new_obj, new_obj_axis = create_vehicle(axis_size=0.15)
    new_obj.transform(key_pose1)
    new_obj_axis.transform(key_pose1)

    # animation 1
    rot_anim = rot_animation(orig_pose, inter_pose, interval=200)
    trans_anim = trans_animation(inter_pose, key_pose1, interval=200)

    animation_display([orig_frame, old_obj, new_obj, new_obj_axis], [rot_anim, trans_anim])

    return
if __name__ == "__main__":
    #main3()
    break_down_rotation()
    #demo_transform_from_origin()

    #dataset = o3d.data.BunnyMesh()
    #obj = o3d.io.read_triangle_mesh(dataset.path)
    #dataset = o3d.data.SwordModel()
    #obj = o3d.io.read_triangle_model(dataset.path)
    #dataset = o3d.data.EaglePointCloud()
    #obj = o3d.io.read_point_cloud(dataset.path)
    #o3d.visualization.draw_geometries([obj])
