#!/usr/bin/env python

import numpy as np
import quaternion
from tf.transformations import euler_from_quaternion
from sklearn.neighbors import NearestNeighbors as KNN

def angvel2mat(w):
    return np.array([[0, -w[0], -w[1], -w[2]],
                    [w[0], 0, w[2], -w[1]],
                    [w[1], -w[2], 0, w[0]],
                    [w[2], w[1], -w[0], 0]])

def euler2rot_matrix (euler):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(euler[0]), -np.sin(euler[0])],
                    [0, np.sin(euler[0]), np.cos(euler[0])]])

    R_y = np.array([[np.cos(euler[1]), 0, np.sin(euler[1])],
                    [0, 1, 0],
                    [-np.sin(euler[1]), 0, np.cos(euler[1])]])

    R_z = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                    [np.sin(euler[2]), np.cos(euler[2]), 0],
                    [0, 0, 1]])

    return R_z.dot(R_y.dot(R_x))

def scan2cart (Z):
    idxs = np.linspace(0,len(Z)-1,60,dtype=int)
    x = np.multiply(Z[idxs,0], np.cos(Z[idxs,1])).reshape(-1)
    y = np.multiply(Z[idxs,0], np.sin(Z[idxs,1])).reshape(-1)
    z = np.zeros(x.shape).reshape(-1)
    output = np.stack((x,y,z)).T
    return output

class ParticleFilter(object):

    def __init__(self, Q_x, Q_q, Map, H,X0, Np = 100):
        self.H = H
        self.Map = Map
        self.Q_x = Q_x
        self.Q_q = Q_q
        self.Np = Np
        self.W = np.ones(Np)/Np
        self.X = np.zeros((Np,3)) + X0
        self.Q = np.zeros((Np,4)) 
        self.euler = np.zeros((Np,3))
        print "initialized!"
        self.nbrs = KNN(n_neighbors=1, algorithm='ball_tree').fit(self.Map)

    def predict (self, Q, dt):
        omega = []
        for ii in range (self.Np):
            self.Q[ii] = Q + np.random.multivariate_normal(np.array([0,0,0,0]), self.Q_q)
            self.euler[ii] = euler_from_quaternion([self.Q[ii][0],self.Q[ii][1],self.Q[ii][2],self.Q[ii][3]])
            self.X[ii] = self.X[ii] + np.random.multivariate_normal(np.array([0,0,0]), self.Q_x)

    def relative_observation(self,Z):
        Z = scan2cart(Z)
        mean_Q = np.mean(self.Q, axis = 0)
        maen_euler = euler_from_quaternion([mean_Q[0],mean_Q[1],mean_Q[2],mean_Q[3]])
        Z_temp = euler2rot_matrix(maen_euler).dot(Z.T) 
        Z_temp = Z_temp.T + np.mean(self.X,axis = 0)
        return Z_temp
            
    def update (self, Z):
        Z = scan2cart(Z)
        for ii in range(self.Np):
            Z_temp =  euler2rot_matrix(self.euler[ii]).dot(Z.T) 
            Z_temp = Z_temp.T + self.X[ii]
            _, indices = self.nbrs.kneighbors(Z_temp[:,0:2])
            Z_star = self.Map[indices].reshape(Z_temp[:,0:2].shape)
            L1 = np.exp(-(1.1)* np.linalg.norm(Z_star-Z_temp[:,0:2],axis=1)**2)
            L2 = np.exp(-(1.1)* (Z_temp[:,2])**2)
            L3 = np.exp(-(1.1)*(Z_temp[:,2]-self.H)**2)
            self.W[ii] = np.sum(0.5*L1+0.25*L2+0.25*L3) 
        self.W = self.W / np.sum(self.W)
        self.resampling()

    def resampling(self):
        index = np.random.choice(a = self.Np,size = self.Np,p = self.W)
        self.W = np.ones (self.Np) / self.Np
        self.X = self.X[index]