from __future__ import division
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


class RobotArm(object):
    def __init__(self):
        self.dh_a=      [     0,      0,     340, 0,     0,    0]
        self.dh_alpha=  [     0,-np.pi/2,    0,     np.pi/2,   -np.pi/2,    np.pi/2]
        self.dh_d=      [ 290,      0,     0,     302,     0, 72] 
        self.dh_offset= [     0,-np.pi/2,    0,     0,         0,    0]
        self.radius=[90, 90, 90, 80, 70, 70, 20]

        self.zone1  = [(-800,-800,-500), (-800, 800,-500), ( 800,-800,-500), (-800,-800, 100)] # ground
        self.zone2  = [(-800,-250, 100), (-800, 250, 100), (-150,-250, 100), (-800,-250, 600)] # front of the robot
        self.zone3a = [(-350, 250, 100), (-350, 450, 100), (-150, 250, 100), (-350, 250, 300)] # container 1
        self.zone3b = [(-350,-450, 100), (-350,-250, 100), (-150,-450, 100), (-350,-450, 300)] # container 2
        
        
    def get_dh_mat(self, a, alpha, d, theta):
        mat = np.array([[ np.cos(theta),            -np.sin(theta),             0,           a ],
                        [ np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
                        [ np.sin(theta)*np.sin(alpha),  np.cos(theta)*np.sin(alpha),  np.cos(alpha),  d*np.cos(alpha)],
                        [0, 0, 0, 1]])

        return mat

    def model(self, angular_positions):
        transforms = np.zeros((4,4,len(self.dh_a)+1))
        T=np.zeros((4,4))
        np.fill_diagonal(T, 1)
        transforms[:,:,0] = T 
        for i, angle in enumerate(angular_positions):
            submat = self.get_dh_mat(self.dh_a[i],self.dh_alpha[i],self.dh_d[i],  self.dh_offset[i] + angle)
            T=np.matmul(T,submat)
            transforms[:,:,i+1] = T 
        return transforms

    def forward_model(self, angular_positions):
        conf=self.model(angular_positions)
        return np.matmul(conf[:,:,-1],np.array([0,0,0,1]))[np.r_[0:3]]
    
    def config_ax(self, ax):
        ax.set_xlim3d(-1000,1000)
        ax.set_ylim3d(-1000,1000)
        ax.set_zlim3d(-1000,1000)
        ax.set_aspect('equal', 'box')
        

    def create_ax(self,fig):
        ax = Axes3D(fig)
        self.config_ax(ax)
        return ax

    def plot_conf(self, ax, angular_positions):
        conf=self.model(angular_positions)
        
        cube_definition = [
            (-100,-100,0), (-100,100,0), (100,-100,0), (-100, -100, 100)
            ]
        self.plot_cube(ax,cube_definition)


        pos = conf[0:3,-1,:]
        
        #self.plot_sphere(ax, [0,0,0])

        for i in range(pos.shape[1]):
            if i==pos.shape[1]-1:
                x=np.matmul( conf[:,:,i], np.array([200,0,0,1]))[np.r_[0:3]]
                y=np.matmul( conf[:,:,i], np.array([0,200,0,1]))[np.r_[0:3]]
                z=np.matmul( conf[:,:,i], np.array([0,0,200,1]))[np.r_[0:3]]
                
                ax.plot([pos[0,i],x[0]],[pos[1,i],x[1]],[pos[2,i],x[2]],'r')
                ax.plot([pos[0,i],y[0]],[pos[1,i],y[1]],[pos[2,i],y[2]],'g')
                ax.plot([pos[0,i],z[0]],[pos[1,i],z[1]],[pos[2,i],z[2]],'b')


            if i>0:
                self.plot_sphere(ax, pos[:,i],1.2*self.radius[i]/2)
                self.plot_cylinder(ax, pos[:,i-1], pos[:,i],self.radius[i]/2)
        
        self.plot_cube(ax,self.zone1,[0.3,0.3,0.3,0.35])
        self.plot_cube(ax,self.zone2,[0.3,0.3,0.8,0.35])
        self.plot_cube(ax,self.zone3a,[0.3,0.8,0.3,0.35])
        self.plot_cube(ax,self.zone3b,[0.3,0.8,0.3,0.35])

    def plot(self, angular_positions):
        fig = plt.figure()
        ax=self.create_ax(fig)
        self.plot_conf(ax,angular_positions)
        plt.show()
        
    def animate(self, angle_init,angle_end, ax = None, predicted_pos=None):

        
        T=100;
        if (ax==None):
            fig = plt.figure()
            ax = self.create_ax(fig)
        for t in range(T):
            ax.clear()
            self.config_ax(ax)
            self.plot_conf(ax,angle_init + t/T * (angle_end-angle_init))
            if(predicted_pos is not None):


                ax.scatter( predicted_pos[0],predicted_pos[1], predicted_pos[2])
            plt.pause(0.01)
        print("end")
        print("predicted:")
        print(predicted_pos)
        print("reached:")
        print(self.forward_model(angle_end))
        return ax
    
    def plot_sphere(self, ax, c=[0, 0, 0], r = 0.05):
        
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
        x = c[0] + r*np.cos(u)*np.sin(v)
        y = c[1] + r*np.sin(u)*np.sin(v)
        z = c[2] + r*np.cos(v)

        ax.plot_surface(x, y, z, color="r")
        
        
    def plot_cylinder(self, ax, origin=np.array([0, 0, 0]), end=np.array([1,1,1]), R = 0.02):
        v = end - origin
        mag = np.linalg.norm(v)
        if mag==0:
            return
        v = v / mag

        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])

        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)

        n2 = np.cross(v, n1)

        t = np.linspace(0, mag, 10)
        theta = np.linspace(0, 2 * np.pi, 10)

        t, theta = np.meshgrid(t, theta)

        X, Y, Z = [origin[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        ax.plot_surface(X, Y, Z,color='orange')


    def plot_cube(self,ax,cube_definition, color=[0.8,0.7,0.3,1]):

        cube_definition_array = [
            np.array(list(item))
            for item in cube_definition
            ]

        points = []
        points += cube_definition_array
        vectors = [
            cube_definition_array[1] - cube_definition_array[0],
            cube_definition_array[2] - cube_definition_array[0],
            cube_definition_array[3] - cube_definition_array[0]
            ]

        points += [cube_definition_array[0] + vectors[0] + vectors[1]]
        points += [cube_definition_array[0] + vectors[0] + vectors[2]]
        points += [cube_definition_array[0] + vectors[1] + vectors[2]]
        points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

        points = np.array(points)

        edges = [
            [points[0], points[3], points[5], points[1]],
            [points[1], points[5], points[7], points[4]],
            [points[4], points[2], points[6], points[7]],
            [points[2], points[6], points[3], points[0]],
            [points[0], points[2], points[4], points[1]],
            [points[3], points[6], points[7], points[5]]
            ]


        faces = Poly3DCollection(edges, linewidths=1)
        faces.set_facecolor(color)

        ax.add_collection3d(faces)
        


