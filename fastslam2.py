import numpy as np
from read_data import read_world, read_sensor_data
from misc_tools import plot_state, angle_diff
import math
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import copy

class Particle(object):

    def __init__(self, num_particles, num_landmarks):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.rmu = np.array([0, 0, 0])
        self.rsigma = np.zeros([3, 3]) # robot location cov
        self.weight = 1.0 / num_particles
        self.history = []
        #  assume motion noise 
        self.P = np.array([[0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.001]])
        self.landmark_dict = {}
        for i in range(num_landmarks):
            landmark = LandMark()
            
            self.landmark_dict[i+1] = landmark
        

class LandMark(object):
    def __init__(self):

        self.mu = np.array([0, 0])
        self.sigma = np.zeros([2, 2])
        self.observed = False
        # sensor noise
        self.R = np.array([[0.1, 0],\
                    [0, 0.1]])
    
    def measurement_model(self, particle):
        #Compute the expected measurement for a landmark
        #and the Jacobian with respect to the landmark.

        px = particle.x
        py = particle.y
        ptheta = particle.theta
        # landmark location
        lx = self.mu[0]
        ly = self.mu[1]

        #calculate expected range measurement
        meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2)
        meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

        z_hat = np.array([meas_range_exp, meas_bearing_exp])

        # Compute the Jacobian H of the measurement function h
        #wrt the landmark location

        H_theta = np.zeros((2,2))
        H_theta[0,0] = (lx - px) / meas_range_exp
        H_theta[0,1] = (ly - py) / meas_range_exp
        H_theta[1,0] = (py - ly) / (meas_range_exp**2)
        H_theta[1,1] = (lx - px) / (meas_range_exp**2)
        H_s = np.array([[-(lx - px) / meas_range_exp, -(ly - py) / meas_range_exp, 0],
                        [(ly - py)/meas_range_exp**2, -(lx - px) / meas_range_exp**2, -1]])


        return z_hat, H_theta, H_s

def motion_model(odometry, particles):
   
    # motion
    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']
    print(delta_rot1, delta_trans, delta_rot2)
    
    for particle in particles:
        #noise free motion
        particle.history.append([particle.x, particle.y])
        particle.x = particle.x + delta_trans*math.cos(particle.theta + delta_rot1)
        particle.y = particle.y + delta_trans*math.sin(particle.theta + delta_rot1)
        particle.theta = particle.theta + delta_rot1 + delta_rot2
        particle.weight = 1.0
        #Jacobian of motion model 
        # G = np.array([[1.0, 0.0, - delta_trans*math.sin(particle.theta + delta_rot1)],
        #             [0.0, 1.0,   delta_trans*math.cos(particle.theta + delta_rot1)],
        #              [0.0, 0.0, 1.0]])

        particle.rmu = np.array([particle.x, particle.y, particle.theta])
        # print("root", G.dot(particle.rsigma).dot(G.T))
        # print(particle.P)
        # G = np.linalg.inv(G)
        #particle.rsigma = (G.dot(particle.rsigma).dot(G.T)) + particle.P
        #particle.rsigma = np.dot(np.dot(G, particle.rsigma),np.transpose(G)) + particle.P
        #print("sigma", particle.rsigma)
        # print(particle.rmu.shape, particle.rsigma.shape)
        # s_hat = np.random.multivariate_normal(particle.rmu, particle.rsigma)
    
        # particle.x, particle.y, particle.theta = s_hat[0], s_hat[1], s_hat[2]
     

    return particles

def multi_normal(x, mean, cov):
    """Calculate the density for a multinormal distribution"""
    den = 2 * math.pi * math.sqrt(scipy.linalg.det(cov))
    num = np.exp(-0.5*np.transpose((x - mean)).dot(scipy.linalg.inv(cov)).dot(x - mean))
    result = num/den
    return result
def eval_sensor_model(particles, sensor_data):
    # sensor measurement
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']


    for particle in particles:
        landmarks = particle.landmark_dict
        
        s_hat = np.array([particle.x, particle.y, particle.theta])
        for i in range(len(ids)):
            lm_id = ids[i]
            print("landmark id", lm_id)
            #print("landmark id", lm_id)
            z = np.array([ranges[i], bearings[i]])
            #print(z)
            landmark = landmarks[lm_id]
            if not landmark.observed:
                lx = particle.x + ranges[i]*math.cos(particle.theta + bearings[i])
                ly = particle.y + ranges[i]*math.sin(particle.theta + bearings[i])
                landmark.mu = np.array([lx, ly]) 
                z_hat, H_theta, H_s = landmark.measurement_model(particle)

                delta_lm = np.array([z[0]-z_hat[0], angle_diff(z[1], z_hat[1])])
                #print(delta_lm)
                Q = H_theta.dot(landmark.sigma).dot(H_theta.T) + landmark.R
                
                #update robot pose
                particle.rsigma = np.linalg.inv(H_s.T.dot(np.linalg.inv(Q)).dot(H_s) + np.linalg.inv(particle.P))
                particle.rmu =  particle.rsigma.dot(H_s.T).dot(np.linalg.inv(Q)).dot(delta_lm) + s_hat
                #print(particle.rmu.shape)
                s = np.random.multivariate_normal(particle.rmu, particle.rsigma)
                particle.x, particle.y, particle.theta = s[0], s[1], s[2]
                #particle.x, particle.y, particle.theta = particle.rmu[0], particle.rmu[1], particle.rmu[2]
                # compute weight 
                # L = H_s.dot(particle.P).dot(H_s.T) + H_theta.dot(landmark.sigma).dot(H_theta.T) + landmark.R
                # weight = stats.multivariate_normal.pdf(delta_lm, mean=np.array([0,0]), cov=L)
                # particle.weight = particle.weight * weight
                # particle.weight = particle.weight * weight
                # update landmark 
                H_theta_inv = np.linalg.inv(H_theta)
                landmark.sigma = H_theta_inv.dot(landmark.R).dot(H_theta_inv.T)
                landmark.observed = True

            else:
                z_hat, H_theta, H_s = landmark.measurement_model(particle)

                delta_lm = np.array([z[0]-z_hat[0], angle_diff(z[1], z_hat[1])])
                Q = H_theta.dot(landmark.sigma).dot(H_theta.T) + landmark.R
                #update robot pose
                particle.rsigma = np.linalg.inv(H_s.T.dot(np.linalg.inv(Q)).dot(H_s) + np.linalg.inv(particle.P))
                particle.rmu =  particle.rsigma.dot(H_s.T).dot(np.linalg.inv(Q)).dot(delta_lm) + s_hat
                #print(particle.rmu.shape)
                # s = np.random.multivariate_normal(particle.rmu, particle.rsigma)
                # particle.x, particle.y, particle.theta = s[0], s[1], s[2]
                particle.x, particle.y, particle.theta = particle.rmu[0], particle.rmu[1], particle.rmu[2]
                # compute weight 
                L = H_s.dot(particle.P).dot(H_s.T) + H_theta.dot(landmark.sigma).dot(H_theta.T) + landmark.R
        
                weight1= stats.multivariate_normal.pdf(delta_lm, mean=np.array([0,0]), cov=L)
                print("weight1", weight1)
                # new update weight
                prior = multi_normal(np.array([particle.x, particle.y, particle.theta]), s_hat, particle.P)
                prop = multi_normal(np.array([particle.x, particle.y, particle.theta]), particle.rmu, particle.rsigma)
                weight2 = prior / prop
                print("weight2",weight2)

                particle.weight = particle.weight * weight1
                #update landmark 
                K = landmark.sigma.dot(H_theta.T).dot(np.linalg.inv(Q))
                landmark.mu = landmark.mu + K.dot(delta_lm)
                print(landmark.mu)
                landmark.sigma = (np.eye(2) - K.dot(H_theta)).dot(landmark.sigma)
        
        
    normalizer = sum([p.weight for p in particles])
    
    for particle in particles:
        #print(particle.weight)
        particle.weight = particle.weight / normalizer
    
    return particles


def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle
    # weights.

    new_particles = []

    '''your code here'''
    '''***        ***'''
    step = 1.0/len(particles)
    u = np.random.uniform(0, step)
    c = particles[0].weight
    i = 0

    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    # new_particle = copy.deepcopy(particles[i])
    # ...
    # new_particles.append(new_particle)

    for particle in particles:

        while u > c:
            i = i + 1
            c = c + particles[i].weight
        new_particle = copy.deepcopy(particles[i])
        new_particle.weight = 1.0/len(particles)
        new_particles.append(new_particle)

        u = u + step

    return new_particles

if __name__ == "__main__":
    plt.axis([-1, 12, 0, 10])
    plt.ion()
    plt.show()

    # load data
    landmarks_gt = read_world("../ais_sim_data/world.dat")
    sensor_data = read_sensor_data("../ais_sim_data/sensor_data.dat")
    # init particles
    num_particles = 200
    num_landmarks = len(landmarks_gt)
    particle_list = []
    for i in range(num_particles):
        particle = Particle(num_particles, num_landmarks)
        particle_list.append(particle)
    for timestep in range(len(sensor_data)//2):
        #print(timestep)
        odometry = sensor_data[timestep,"odometry"]
        sensor_measurement = sensor_data[timestep, "sensor"]
        particle_list = motion_model(odometry, particle_list)
        particle_list = eval_sensor_model(particle_list, sensor_measurement)
        plot_state(particle_list, landmarks_gt)

        particle_list = resample_particles(particle_list)

    plt.show()






    





