import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import open3d.visualization as vis
import math
from numpy import linalg as LA
import time
import copy
from itertools import chain

def get_axis():
    points = [
                [0, 0, 0],
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5],
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

def create_vehicle():
    body_frame = get_axis()
    body_frame.translate(np.array([2,0,0]))
    obj = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.15,create_uv_map=True)
    obj.translate(np.array([2.0,0,0.1]))
    return obj, body_frame


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
    T = Rotation.from_quat(quat).as_matrix()
    T[0,3] = position[0]
    T[1,3] = position[1]
    T[2,3] = position[2]
    return T


def main():

    rot1, rot1_3x3 = get_rotation_matrix(45, 45, 45)
    rot2, rot2_3x3 = get_rotation_matrix(90, 45, 30)
    T = get_homo_translate_matrix(1,1,1)

    rot2_part1, _ = get_rotation_matrix(0,0,30)
    rot2_part2, _ = get_rotation_matrix(0,45,0)
    rot2_part3, _ = get_rotation_matrix(90,0,0)
    concate = np.dot(rot2_part1, np.dot(rot2_part2, rot2_part3))
    #print(concate)
    #print(rot2)

    rot2_forward1 = rot2_part3
    rot2_forward2 = np.dot(rot2_part2, rot2_forward1)
    rot2_forward3 = np.dot(rot2_part1, rot2_forward2)
    #print(rot2_forward3)

    inv_z, _ = get_ele_rotation_matrix('z',-90)
    inv_y, _ = get_ele_rotation_matrix('y', -45)
    inv_x, _ = get_ele_rotation_matrix('x', -30)

    rot2_back1 = np.dot(inv_x, rot2)
    rot2_back2 = np.dot(inv_y, rot2_back1)
    rot2_back3 = np.dot(inv_z, rot2_back2)
    print(rot2_back3)
    
    orig_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

    obj, obj_axis = create_vehicle()
    #obj.transform(rot2)
    #obj_axis.transform(rot2)
    #o3d.visualization.draw_geometries([orig_frame, obj, obj_axis])

    qrot = Rotation.from_matrix(rot2_3x3).as_quat()
    qrot2 = Rotation.from_matrix(rot1_3x3).as_quat()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(orig_frame)
    vis.add_geometry(obj)
    vis.add_geometry(obj_axis)
    end_obj = copy.deepcopy(obj)
    end_obj_axis = copy.deepcopy(obj_axis)
    end_obj.transform(rot2)
    end_obj_axis.transform(rot2)
    vis.add_geometry(end_obj)
    vis.add_geometry(end_obj_axis)
    vis.poll_events()
    vis.update_renderer()
    
    #cos = qrot[0]
    #sin = LA.norm(qrot[1:])
    #angle = math.atan(sin/cos)
    #print('full angle is', angle * 180 / np.pi)
    #rot_axis = qrot[1:] / sin
    ##print(LA.norm(qrot), sum(i*i for i in qrot[1:]), qrot[0]*qrot[0])


    def slerp(quat, interval=100):
        # calc end quat angle and axis
        print('original quat:', quat)
        cos = qrot[3]
        sin = LA.norm(qrot[0:3])
        angle = math.atan(sin/cos)
        print('full angle is', angle * 180 / np.pi)
        rot_axis = qrot[0:3] / sin

        for i in range(0, interval):
            angle_step = angle *i /interval 
            print('angle step=', angle_step * 180 / np.pi)
            qrot_step = np.zeros((4,))
            qrot_step[3] = math.cos(angle_step)
            qrot_step[0:3] = rot_axis * math.sin(angle_step)
            #print(LA.norm(qrot_step))
            rot_step_matrix, _ = get_homo_quaternion_matrix(qrot_step)
            print(i)
            print(qrot_step)
            print(rot_step_matrix)
            yield rot_step_matrix

    '''
    ref: https://dl.acm.org/doi/pdf/10.1145/325334.325242
    '''
    def another_slerp(quat1, quat2, interval=100):
        quat1 = np.array(quat1)
        quat2 = np.array(quat2)
        cos_theta = np.inner(quat1, quat2) 
        sin_theta = math.sqrt(1-cos_theta*cos_theta)
        theta = math.atan(sin_theta/cos_theta)

        for u in np.arange(0, 1, float(1./interval)):
            new_quat = math.sin(theta*(1.-u))/sin_theta * quat1 + math.sin(theta*u)/sin_theta * quat2
            rot_mat, _ = get_homo_quaternion_matrix(new_quat)
            yield rot_mat 

    

    #interpolation_tf_iterator = slerp(qrot, 100)
    #interpolation_tf_iterator_1 = another_slerp([0,0,0,1], qrot, 100)
    #interpolation_tf_iterator_2 = another_slerp(qrot, qrot2, 100)
    qrot_forward1 = Rotation.from_matrix(rot2_forward1[:3,:3]).as_quat()
    qrot_forward2 = Rotation.from_matrix(rot2_forward2[:3,:3]).as_quat()
    qrot_forward3 = Rotation.from_matrix(rot2_forward3[:3,:3]).as_quat()
    qrot_back1 = Rotation.from_matrix(rot2_back1[:3,:3]).as_quat()
    qrot_back2 = Rotation.from_matrix(rot2_back2[:3,:3]).as_quat()
    qrot_back3 = Rotation.from_matrix(rot2_back3[:3,:3]).as_quat()

    forward_tf_1 = another_slerp([0,0,0,1], qrot_forward1)
    forward_tf_2 = another_slerp(qrot_forward1, qrot_forward2)
    forward_tf_3 = another_slerp(qrot_forward2, qrot_forward3)
    reverse_tf_1 = another_slerp(qrot, qrot_back1)
    reverse_tf_2 = another_slerp(qrot_back1, qrot_back2)
    reverse_tf_3 = another_slerp(qrot_back2, qrot_back3)
    #all_iter = chain(interpolation_tf_iterator_1, reverse_tf_1, reverse_tf_2, reverse_tf_3)
    all_iter = chain(forward_tf_1, forward_tf_2, forward_tf_3, reverse_tf_1, reverse_tf_2, reverse_tf_3)

    for i in range(600):
        rot_step_matrix = next(all_iter)
        #input()
        obj.transform(rot_step_matrix)
        obj_axis.transform(rot_step_matrix)
        vis.update_geometry(obj)
        vis.update_geometry(obj_axis)
        #vis.update_geometry(orig_frame)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        #orig_frame.transform(np.linalg.inv(rot_step_matrix))
        obj.transform(np.linalg.inv(rot_step_matrix))
        obj_axis.transform(np.linalg.inv(rot_step_matrix))
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    return


    straight_route = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([[-10, -10, -10], [10, 10, 10]])
            )

if __name__ == "__main__":
    main()
