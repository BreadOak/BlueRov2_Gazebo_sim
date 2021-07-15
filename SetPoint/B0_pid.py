from scipy import linalg
from scipy.integrate import solve_ivp
from pymavlink import mavutil
import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
import scipy

## Control frequency is defined as follows
s_freq = float(100)
s_time = 1/s_freq

# ## End time
# Tf = 10

## BlueRov2 parameter
m = 10     # kg
W = 98.1    # N
B = 100.6   # N
Ix = 0.16   # kg*m^2
Iy = 0.16   # kg*m^2
Iz = 0.16   # kg*m^2
# Ix = 1   # kg*m^2
# Iy = 1   # kg*m^2
# Iz = 1   # kg*m^2

rg = np.array([0, 0, 0])    # m
rb = np.array([0, 0, 0.02]) # m
BG = rg - rb      # m
xg = BG[0]        # m
yg = BG[1]        # m
zg = BG[2]        # m

Xud = -5.5   # kg
Yvd = -12.7  # kg
Zwd = -14.57 # kg
Kpd = -0.12  # kg*m^2/rad
Mqd = -0.12  # kg*m^2/rad
Nrd = -0.12  # kg*m^2/rad

Xu = -4.03   # N*s/m
Yv = -6.22   # N*s/m
Zw = -5.18   # N*s/m
Kp = -0.07   # N*s/rad
Mq = -0.07   # N*s/rad
Nr = -0.07   # N*s/rad

Xuu = -18.18 # N*s^2/m^2
Yvv = -21.66 # N*s^2/m^2
Zww = -36.99 # N*s^2/m^2
Kpp = -1.55  # N*s^2/rad^2
Mqq = -1.55  # N*s^2/rad^2
Nrr = -1.55  # N*s^2/rad^2

## PID Gain
KP = np.array([[0.1], [0.1], [0.05], [0.05]])
KI = np.array([[0], [0], [0], [0]])
KD = np.array([[0], [0], [0], [0]])

def Setpoint_PID(master, Des_x, Des_y, Des_z, Des_ps):

	global J, F, M, C, D, g

	# Initial & Desired state
	Initial_state = np.array([ [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0] ],dtype = float)
	Desired_state = np.array([ [Des_x], [Des_y], [Des_z], [0], [0], [Des_ps], [0], [0], [0], [0], [0], [0] ],dtype = float)
	Current_state = Initial_state

	# Initial Previous error & Icontrol gain
	IControl = 0
	prev_E = np.zeros((4,1))

	while(1):
	#for i in range(0,1) :

		x  = float(Current_state[0])
		y  = float(Current_state[1])
		z  = float(Current_state[2])
		ph = float(Current_state[3])
		th = float(Current_state[4])
		ps = float(Current_state[5])
		u  = float(Current_state[6])
		v  = float(Current_state[7])
		w  = float(Current_state[8]) 
		p  = float(Current_state[9])
		q  = float(Current_state[10])
		r  = float(Current_state[11])

		# Set Error
		Error = Desired_state - Current_state 
		Error_X_world = Error[0:6]
		Error_V_body  = Error[6:12]

		E = np.vstack((Error_X_world[0:3],Error_X_world[5]))
		print('E',E)

		# Transformation Matrix of linear velocity
		R = np.array([[math.cos(ps)*math.cos(th), -math.sin(ps)*math.cos(ph)+math.sin(ph)*math.sin(th)*math.cos(ps),  math.sin(ps)*math.sin(ph)+math.sin(th)*math.cos(ps)*math.cos(ph)],
		              [math.sin(ps)*math.cos(th),  math.cos(ps)*math.cos(ph)+math.sin(ph)*math.sin(th)*math.sin(ps), -math.cos(ps)*math.sin(ph)+math.sin(th)*math.sin(ps)*math.cos(ph)],
		              [            -math.sin(th),                                         math.sin(ph)*math.cos(th),                                         math.cos(ph)*math.cos(th)]])

		# Transformation Matrix of angular velocity
		T = np.array([[1,  math.sin(ph)*math.tan(th),  math.cos(ph)*math.tan(th)],
			          [0,               math.cos(ph),              -math.sin(ph)],
			          [0,  math.sin(ph)/math.cos(th),  math.cos(ph)/math.cos(th)]])

		# Transformation Matrix (Body -> World)
		J = np.hstack([np.vstack([R,np.zeros((3,3))]), np.vstack([np.zeros((3,3)),T])])

		# Mass Matrix
		Mrb = np.array([[   m,      0,      0,      0,   m*zg,      0],
	                    [   0,      m,      0,  -m*zg,      0,      0],
	                    [   0,      0,      m,      0,      0,      0],
	                    [   0,  -m*zg,      0,     Ix,      0,      0],
	                    [m*zg,      0,      0,      0,     Iy,      0],
	                    [   0,      0,      0,      0,      0,     Iz]])

		Ma  = - np.array([[Xud,   0,   0,   0,   0,   0],
	                      [  0, Yvd,   0,   0,   0,   0],
	                      [  0,   0, Zwd,   0,   0,   0],
	                      [  0,   0,   0, Kpd,   0,   0],
	                      [  0,   0,   0,   0, Mqd,   0],
	                      [  0,   0,   0,   0,   0, Nrd]])

		M = Mrb + Ma

		# Coriolis force Matrix
		Crb_L = np.array([[   0,     0,     0,       0,     m*w,    -m*v],
	                      [   0,     0,     0,    -m*w,       0,     m*u],
	                      [   0,     0,     0,     m*v,    -m*u,       0],
	                      [   0,   m*w,  -m*v,       0,    Iz*r,   -Iy*q],
	                      [-m*w,     0,   m*u,   -Iz*r,       0,    Ix*p],
	                      [ m*v,  -m*u,     0,    Iy*q,   -Ix*p,       0]])

		Crb_V = np.array([[   0,  -m*r,   m*q,       0,       0,       0],
	                      [ m*r,     0,  -m*p,       0,       0,       0],
	                      [-m*q,   m*p,     0,       0,       0,       0],
	                      [   0,     0,     0,       0,    Iz*r,   -Iy*q],
	                      [   0,     0,     0,   -Iz*r,       0,    Ix*p],
	                      [   0,     0,     0,    Iy*q,   -Ix*p,       0]])

		Ca = np.array([[     0,       0,      0,       0,   -Zwd*w,  Yvd*v],
	                   [     0,       0,      0,   Zwd*w,        0, -Xud*u],
	                   [     0,       0,      0,  -Yvd*v,    Xud*u,      0],
	                   [     0,  -Zwd*w,  Yvd*v,       0,   -Nrd*r,  Mqd*q],
	                   [ Zwd*w,       0, -Xud*u,   Nrd*r,        0, -Kpd*p],
	                   [-Yvd*v,   Xud*u,      0,  -Mqd*q,    Kpd*p,      0]])

		C = Crb_L + Ca

		# Damping Matrix
		D = -np.array([[ Xu + Xuu*abs(u),                0,                 0,                 0,                 0,                 0],
	                   [               0,  Yv + Yvv*abs(v),                 0,                 0,                 0,                 0],
	                   [               0,                0,   Zw + Zww*abs(w),                 0,                 0,                 0],
	                   [               0,                0,                 0,   Kp + Kpp*abs(p),                 0,                 0],
	                   [               0,                0,                 0,                 0,   Mq + Mqq*abs(q),                 0],
	                   [               0,                0,                 0,                 0,                 0,   Nr + Nrr*abs(r)]])

		# Resorting Force
		g = np.array([[ (W - B) * math.sin(th)             ],
			          [-(W - B) * math.cos(th)*math.sin(ph)],
			          [-(W - B) * math.cos(th)*math.cos(ph)],
			          [  zg * W * math.cos(th)*math.sin(ph)],
			          [  zg * W * math.sin(th)             ],
			          [                                   0]])

		PControl = KP * E
		#IControl = IControl + KI * E * s_time  
		DControl = KD * (E - prev_E)/s_time 

		# Control Inputs between [-1 and 1]
		U = PControl + IControl + DControl
		#print('U',U)
		if abs(U[0]) > 1 :
			U[0] = 1*(U[0])/abs(U[0])
		if abs(U[1]) > 1 :
			U[1] = 1*(U[1])/abs(U[1])
		if abs(U[2]) > 1 :
			U[2] = 1*(U[2])/abs(U[2])
		if abs(U[3]) > 1 :
			U[3] = 1*(U[3])/abs(U[3])

		# x,y,ps between [-1000 and 1000], z between [0 and 1000]
		master.mav.manual_control_send(master.target_system, U[0]*1000, U[1]*1000, U[2]*500+500, U[3]*1000, 0)

		# Allocation Matrix
		Al = np.array([[-1, 1, 0, 1],
			           [-1,-1, 0,-1],
			           [ 1, 1, 0,-1],
			           [ 1,-1, 0, 1],
			           [ 0, 0,-1, 0],
			           [ 0, 0,-1, 0]])

		# Allocated Control Signal
		a = np.dot(Al,U)

		# Thrust Vector
		t = Get_ThrustVector(a)

		# Propulsion Matrix
		K = np.array([[ 0.7071, 0.7071, -0.7071, -0.7071,     0,      0],
			          [-0.7071, 0.7071, -0.7071,  0.7071,     0,      0],
			          [      0,      0,       0,       0,    -1,     -1],
			          [      0,      0,       0,       0, 0.115, -0.115],
			          [      0,      0,       0,       0,     0,      0],
			          [-0.1773, 0.1773,  0.1773, -0.1773,     0,      0]])

		# K = np.array([[ 0.707, -0.707,  0.707, -0.707,      0,      0],
		# 	          [ 0.707,  0.707, -0.707, -0.707,      0,      0],
		# 	          [     0,      0,      0,      0,     -1,     -1],
		# 	          [     0,      0,      0,      0,      0,      0],
		# 	          [     0,      0,      0,      0, -0.111,  0.111],
		# 	          [-0.167,  0.167,  0.167, -0.167,      0,      0]])

		# Total Force and Moment
		F = np.dot(K,t)
		print('F',F)

		# Current State
		X = np.array([ [x], [y], [z], [ph], [th], [ps] ]) # World frame
		V = np.array([ [u], [v], [w],  [p],  [q],  [r] ]) # Body frame

		# Runge-Kutta 4
		kx1 = dxdt1(t,X,V)
		kv1 = dxdt2(t,X,V)

		kx2 = dxdt1(t + s_time/2, X + s_time*kx1/2, V + s_time*kv1/2 )
		kv2 = dxdt2(t + s_time/2, X + s_time*kx1/2, V + s_time*kv1/2 )

		kx3 = dxdt1(t + s_time/2, X + s_time*kx2/2, V + s_time*kv2/2 )
		kv3 = dxdt2(t + s_time/2, X + s_time*kx2/2, V + s_time*kv2/2 )

		kx4 = dxdt1(t + s_time, X + s_time*kx3, V + s_time*kv3 )
		kv4 = dxdt2(t + s_time, X + s_time*kx3, V + s_time*kv3 )
		
		dx = s_time*(kx1 + 2*kx2 + 2*kx3 + kx4)/6
		dv = s_time*(kv1 + 2*kv2 + 2*kv3 + kv4)/6

	    # Next State
		X = X + dx
		V = V + dv

		# Update state & Set Previous Error
		Current_state = np.vstack([X,V])
		prev_E = E

		print('X:',Current_state[0]),
		print('Y:',Current_state[1]),
		print('Z:',Current_state[2])
		# print('ps:',Current_state[5])

def dxdt1(t,X,V):
	global J
	dxdt = np.dot(J,V)
	return dxdt 

def dxdt2(t,X,V):
	global F, M, C, D, g
	dvdt = np.dot(linalg.inv(M) , (F-np.dot(C,V)-np.dot(D,V)-g))
	return dvdt 

def Get_ThrustVector(a):
	t1 = 80/(1+np.exp(-4*a[0]**3)) - 40
	t2 = 80/(1+np.exp(-4*a[1]**3)) - 40
	t3 = 80/(1+np.exp(-4*a[2]**3)) - 40
	t4 = 80/(1+np.exp(-4*a[3]**3)) - 40
	t5 = 80/(1+np.exp(-4*a[4]**3)) - 40
	t6 = 80/(1+np.exp(-4*a[5]**3)) - 40
	t = np.array([t1, t2, t3, t4, t5, t6])
	return t

if __name__ =='__main__':

	## connect
	master_0 = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
	master_0.wait_heartbeat()

	## Arm
	master_0.mav.command_long_send(master_0.target_system, master_0.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
	Setpoint_PID(master_0, 0, 0, 0, 0)
	#master_0.mav.manual_control_send(master_0.target_system, 500, 500, 500, 0, 0)