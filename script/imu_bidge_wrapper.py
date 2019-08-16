#!/usr/bin/env python

import rospy
import numpy as np 
from sensor_msgs.msg import Imu, LaserScan, PointCloud
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped, Quaternion, Point
from nav_msgs.srv import GetMap
from particle_filter import ParticleFilter
from px_comm.msg import OpticalFlow
from tf import TransformBroadcaster

rospy.init_node('imu_bridge_wrapper', anonymous = True)
last_time_imu = rospy.Time.now().to_sec()
last_time_px4flow = rospy.Time.now().to_sec()

first_run = True
Vxy = []
W = []

def imu_callback(msg):
    global last_time_imu, partical_filter, first_run, a_ref, W, Vxy
    if first_run is False:
        imu = msg
        time = imu.header.stamp.secs + imu.header.stamp.nsecs * 10e-9
        dt = time - last_time_imu
        last_time_imu = time
        W = [imu.angular_velocity.x , imu.angular_velocity.y, imu.angular_velocity.z]
        partical_filter.predict (W,np.array([0,0,0]),dt)
        
def pixelflow_callback(msg):
    global last_time_px4flow, partical_filter, first_run, a_ref, W, Vxy
    if first_run is False:
        pixflow = msg
        time = pixflow.header.stamp.secs + pixflow.header.stamp.nsecs * 10e-9
        dt = time - last_time_px4flow
        last_time_px4flow = time
        Vxy = np.array([pixflow.velocity_x , pixflow.velocity_y, 0.0])
        print "recived pixel flow data: "+str(Vxy)
        partical_filter.predict ([0,0,0],Vxy,dt)

def scan_callback(msg):
    global partical_filter, first_run
    if first_run is False:
        scan = convert_scan(msg)
        partical_filter.update(scan)
        send_other_robot_observation(msg)

def convert_scan(scan):
    global map_info
    theta_reset = map_info.origin.orientation.z
    r = scan.ranges
    i = np.arange(len(r))
    theta = np.add(scan.angle_min, np.multiply(i, scan.angle_increment)) + theta_reset
    theta = np.delete(theta,np.where(~np.isfinite(r)),axis=0)
    r = np.delete(r,np.where(~np.isfinite(r)),axis=0)
    return np.vstack((r,np.flip(theta,-1))).T

def occupancy_grid():
    static_map = rospy.ServiceProxy('static_map',GetMap)
    map = static_map()
    return np.transpose(np.array(map.map.data).reshape(np.array(map.map.info.width), np.array(map.map.info.height)))

def obs():
    static_map = rospy.ServiceProxy('static_map',GetMap)
    map = static_map()
    occupancy_grid = np.transpose(np.array(map.map.data).reshape(np.array(map.map.info.width), np.array(map.map.info.height)))
    return np.argwhere(occupancy_grid == 100) * map.map.info.resolution + map.map.info.resolution * 0.5 + map.map.info.origin.position.x

def pub2rviz ():
    global pub_particlecloud, pub_estimated_pos
    laser_tf_br = TransformBroadcaster()
    particle_pose = PoseArray()
    particle_pose.header.frame_id = 'map'
    particle_pose.header.stamp = rospy.Time.now()
    particle_pose.poses = []

    estimated_pose = PoseWithCovarianceStamped()
    estimated_pose.header.frame_id = 'map'
    estimated_pose.header.stamp = rospy.Time.now()
    estimated_pose.pose.pose.position.x = np.mean(partical_filter.X[:,0])
    estimated_pose.pose.pose.position.y = np.mean(partical_filter.X[:,1])
    estimated_pose.pose.pose.position.z = np.mean(partical_filter.X[:,2])
    quaternion = np.mean(partical_filter.Q, axis=0)
    quaternion = normalize_quaternion (quaternion)
    estimated_pose.pose.pose.orientation = Quaternion(*quaternion)

    for ii in range(partical_filter.Np):
        pose = Pose()
        point_P = (partical_filter.X[ii,0],partical_filter.X[ii,1],partical_filter.X[ii,2])
        pose.position = Point(*point_P)

        quaternion = partical_filter.Q[ii]
        quaternion = normalize_quaternion (quaternion)
        pose.orientation = Quaternion(*quaternion)

        particle_pose.poses.append(pose)

    pub_particlecloud.publish(particle_pose)
    pub_estimated_pos.publish(estimated_pose)
    laser_tf_br.sendTransform((np.mean(partical_filter.X[:,0]) , np.mean(partical_filter.X[:,1]) , np.mean(partical_filter.X[:,2])),
                        (estimated_pose.pose.pose.orientation.w,estimated_pose.pose.pose.orientation.x,estimated_pose.pose.pose.orientation.y,estimated_pose.pose.pose.orientation.z),
                        rospy.Time.now(),
                        "laser_frame",
                        "map")

def send_other_robot_observation(msg):
        global partical_filter, observation_pub
        particle_pose = PointCloud()
        particle_pose.header.frame_id = 'map'
        particle_pose.header.stamp = rospy.Time.now()
        particle_pose.points = []
        Z = convert_scan(msg)
        z = partical_filter.relative_observation(Z)
        for ii in range(len(z)):
                point = Point()
                point.x = z[ii,0]
                point.y = z[ii,1]
                point.z = z[ii,2]
                
                particle_pose.points.append(point)

        observation_pub.publish(particle_pose)

def normalize_quaternion(quaternion):
        quat = np.array(quaternion)
        return quat / np.sqrt(np.dot(quat, quat))

def main():
    global map_info, partical_filter, first_run, pub_particlecloud, pub_estimated_pos, a_ref, observation_pub
    static_map = rospy.ServiceProxy('static_map',GetMap)
    map_info = static_map().map.info
    #occupancy_grid = imu_bidge_wrapper.occupancy_grid()
    objects_map  = obs()
    rospy.Subscriber('px4flow/opt_flow', OpticalFlow, pixelflow_callback,queue_size = 1)
    rospy.Subscriber('imu', Imu, imu_callback,queue_size = 1)
    REF = imu_first = rospy.wait_for_message('imu', Imu)
    rospy.Subscriber('scan', LaserScan, scan_callback, queue_size = 1)
    rospy.wait_for_message('scan', LaserScan)
    pub_particlecloud = rospy.Publisher('/particlecloud', PoseArray, queue_size = 10)
    pub_estimated_pos = rospy.Publisher('/estimated_pose', PoseWithCovarianceStamped, queue_size = 10)
    observation_pub = rospy.Publisher('/observation', PointCloud, queue_size = 10)

    a_ref = [REF.linear_acceleration.x , REF.linear_acceleration.y, REF.linear_acceleration.z]

    R_w = np.diag([0.01,0.01,0.01])
    Q_x = np.diag([0.01,0.01,0.05])
    Q_v = np.diag([0.01,0.01,0.05])
    Q_q = 0.01 * np.eye(4)
    X0 = np.array([0.0,0.0,0.8])
    partical_filter = ParticleFilter(R_w, Q_x, Q_v,  Q_q, objects_map, H = 4, X0= X0,  Np=50)
    first_run = False
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        pub2rviz()   
        rate.sleep()

    rospy.spin()


if __name__ == "__main__":
    main()