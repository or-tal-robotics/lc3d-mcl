#!/usr/bin/env python

import numpy as np
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
    x = np.multiply(Z[:,0], np.cos(Z[:,1])).reshape(-1)
    y = np.multiply(Z[:,0], np.sin(Z[:,1])).reshape(-1)
    z = np.zeros(x.shape).reshape(-1)
    output = np.stack((x,y,z)).T
    return output

class ParticleFilter(object):

    def __init__(self, R_w, Q_x, Q_v, Q_q, Map, H, Np = 100):
        self.H = H
        self.Map = Map
        self.R_w = R_w
        self.Q_x = Q_x
        self.Q_v = Q_v
        self.Q_q = Q_q
        self.Np = Np
        self.W = np.ones(Np)/Np
        self.X = np.zeros((Np,3))
        self.V = np.zeros((Np,3))
        self.Q = np.zeros((Np,4))
        self.euler = np.zeros((Np,3))
        print "initialized!"
        self.nbrs = KNN(n_neighbors=1, algorithm='ball_tree').fit(self.Map)

    def predict (self, w, v, dt):
        for ii in range (self.Np):
            wt = w + np.random.multivariate_normal(np.array([0,0,0]), dt*self.R_w)
            self.Q[ii] = (0.5 * angvel2mat(wt) * dt +np.eye(4)).dot(self.Q[ii])+ np.random.multivariate_normal(np.array([0,0,0,0]), dt*self.Q_q)
            self.euler[ii] = euler_from_quaternion([self.Q[ii][0],self.Q[ii][1],self.Q[ii][2],self.Q[ii][3]])
            vi = v + np.random.multivariate_normal(np.array([0,0,0]), dt*self.Q_v)
            self.X[ii] = self.X[ii] + vi * dt + np.random.multivariate_normal(np.array([0,0,0]), dt*self.Q_x)
            
    def update (self, Z):
        Z = scan2cart(Z)
        for ii in range(self.Np):
            Z_temp =  euler2rot_matrix(self.euler[ii]).dot(Z.T) 
            Z_temp = Z_temp.T
            _, indices = self.nbrs.kneighbors(Z_temp[:,0:2])
            Z_star = self.Map[indices].reshape(Z_temp[:,0:2].shape)
            L1 = np.exp(-(1.1)* np.linalg.norm(Z_star-Z_temp[:,0:2],axis=1)**2)
            L2 = np.exp(-(1.1)* (Z_temp[:,2])**2)
            L3 = np.exp(-(1.1)*(Z_temp[:,2]-self.H)**2)
            self.W[ii] = np.prod(0.5*L1+0.25*L2+0.25*L3) 
        self.W = self.W / np.sum(self.W)
        self.resampling()

    def resampling(self):
        index = np.random.choice(a = self.Np,size = self.Np,p = self.W)
        self.W = np.ones (self.Np) / self.Np
        self.Q = self.Q[index]
        self.X = self.X[index]