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
        self.weight = 1.0 / num_particles
        self.history = []

        self.landmark_dict = {}
        for i in range(num_landmarks):
            landmark = LandMark()
            
            self.landmark_dict[i+1] = landmark
        

class LandMark(object):
    def __init__(self):

        self.mu = [0, 0]
        self.sigma = np.zeros([2, 2])
        self.observed = False
        #sensor noise
        self.R = np.array([[0.1, 0],\
                    [0, 0.1]])

    def update(self, particle, real_meas):
        px = particle.x
        py = particle.y
        ptheta = particle.theta
        meas_range, meas_bearing = real_meas[0], real_meas[1]
        if not self.observed:
                # landmark is observed for the first time

                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
  
                lx = px + meas_range * np.cos(ptheta + meas_bearing)
                ly = py + meas_range * np.sin(ptheta + meas_bearing)
                self.mu = [lx, ly]
                h, H = self.measurement_model(particle)
                H_inv = np.linalg.inv(H)
                self.sigma = H_inv.dot(self.R).dot(H_inv.T)

                self.observed = True

        else:
            # landmark was observed before

            # update landmark mean and covariance. You can use the
            # provided function 'measurement_model' above.
            # calculate particle weight: particle['weight'] = ...
            h, H = self.measurement_model(particle)
            S = self.sigma
            Q = H.dot(S).dot(H.T) + self.R
            K = S.dot(H.T).dot(np.linalg.inv(Q))
           
            delta = np.array([meas_range - h[0], angle_diff(meas_bearing, h[1])])

            self.mu = self.mu + K.dot(delta)
            self.sigma = (np.identity(2) - K.dot(H)).dot(S)

            weight = stats.multivariate_normal.pdf(delta, mean=np.array([0,0]), cov=Q)
            particle.weight = particle.weight * weight
    
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

        h = np.array([meas_range_exp, meas_bearing_exp])

        # Compute the Jacobian H of the measurement function h
        #wrt the landmark location

        H = np.zeros((2,2))
        H[0,0] = (lx - px) / h[0]
        H[0,1] = (ly - py) / h[0]
        H[1,0] = (py - ly) / (h[0]**2)
        H[1,1] = (lx - px) / (h[0]**2)

        return h, H

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.01, 0.01, 0.005, 0.005]

    '''your code here'''
    '''***        ***'''
    #
    delta_hat_r1 = delta_rot1 + np.random.normal(0, noise[0]*abs(delta_rot1) + noise[1]*delta_trans)
    delta_hat_r2 = delta_rot2 + np.random.normal(0, noise[0]*abs(delta_rot2) + noise[1]*delta_trans)
    delta_hat_t = delta_trans + np.random.normal(0, noise[2]*delta_trans + noise[3]*(abs(delta_rot1) + abs(delta_rot2)))
    for particle in particles:
        #add motion model here
        #particle['x'] = ....
        particle.history.append([particle.x, particle.y])
        particle.x = particle.x + delta_hat_t*math.cos(particle.theta + delta_hat_r1)
        particle.y = particle.y + delta_hat_t*math.sin(particle.theta + delta_hat_r1)
        particle.theta = particle.theta + delta_hat_r1 + delta_hat_r2
    return particles


def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle.landmark_dict
        particle.weight = 1.0

        #loop over observed landmarks
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]


            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            real_meas = [meas_range, meas_bearing]

            landmark.update(particle, real_meas)


    #normalize weights
    normalizer = sum([p.weight for p in particles])

    for particle in particles:
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

def data_association():
    
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
        print(timestep)
        odometry = sensor_data[timestep,"odometry"]
        particle_list = sample_motion_model(odometry, particle_list)
        sensor_measurement = sensor_data[timestep, "sensor"]
        particle_list = eval_sensor_model(sensor_measurement, particle_list)
        plot_state(particle_list, landmarks_gt)

        particle_list = resample_particles(particle_list)

    plt.show()






    





