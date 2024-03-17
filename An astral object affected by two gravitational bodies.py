# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 23:46:05 2022

@author: tommi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
'''
def function for use in solve_ivp method

def f1234(t,state,G,M):
    x,y,vx,vy = state
    dxdt = vx
    dydt = vy
    dvxdt = (-G*M*x/(x**2+y**2)**(3/2))
    dvydt = (-G*M*y/(x**2+y**2)**(3/2))
    return (dxdt,dydt,dvxdt,dvydt)
'''

def f1(vx):
    return (vx)

def f2(vy):
    return (vy)
    
def f3(x,y):
    return (-G*M*x/((x**2 + y**2)**(3/2)))

def f4(x,y):
    return (-G*M*y/((x**2 + y**2)**(3/2)))

G=6.67e-11 #gravitational constant
M=5.97e24 #mass of earth
t0=0 #intial time 
tmax=8.5*86400 # time in seconds where 86400 is one day (can be altered to view longer trajectory)
numpoints = 100000

''' 
Setting intial conditions
The commented initial conditions become inputs by user later in the code.

'''

#x0= 6.67e6 + 6e6 # radius of earth + distance above earth's surface
#y0 = 0 
#vx0 = 0
#vy0 = 7000 
#x[0] = x0
#y[0] = y0
#vx[0] = vx0
#vy[0] = vy0 

Myinput = '0'
while Myinput != 'q':
    Myinput = input('Enter a choice, "a", "b" or "q" to quit: ')
    if Myinput == 'a':
        print('You chose part',Myinput)
        t=np.linspace(t0,tmax,numpoints)
        dt=(tmax-t0)/(numpoints-1)
        x=np.zeros(numpoints)
        y=np.zeros(numpoints)
        vx=np.zeros(numpoints)
        vy=np.zeros(numpoints)
        t[0] = t0
        i=0
        h = dt
        
        Inputx = input('Enter a value for x (in meters greater than |6e6|): ')
        x0 = float(Inputx)
        x[0] = x0
        Inputy = input('Enter a value for y (in meters greater than |6e6|): ')
        y0 = float(Inputy)
        y[0] = y0
        Inputvx = input('Enter a value for vx (in m/s): ')
        vx0 = Inputvx
        vx[0] = vx0
        Inputvy = input('Enter a value for vy (in m/s): ')
        vy0 = Inputvy
        vy[0] = vy0
            
        while t[i] < tmax and ((x[i])**2 + (y[i])**2)**0.5 > 6.67e6: 
            #the extra condition stops the simulation when the orbit has crashed
            k1x = f1(vx[i])
            k1y = f2(vy[i])
            k1vx = f3(x[i],y[i])
            k1vy = f4(x[i],y[i])
            
            k2x = f1(vx[i]+(h*k1vx/2))
            k2y = f2(vy[i]+(h*k1vy/2))
            k2vx = f3((x[i]+(h*k1x/2)),(y[i]+(h*k1y/2)))
            k2vy = f4((x[i]+(h*k1x/3)),(y[i]+(h*k1y/2)))
            
            k3x = f1(vx[i]+(h*k2vx/2))
            k3y = f2(vy[i]+(h*k2vy/2))
            k3vx = f3((x[i]+(h*k2x/2)),(y[i]+(h*k2y/2)))
            k3vy = f4((x[i]+(h*k2x/2)),(y[i]+(h*k2y/2)))
            
            k4x = f1(vx[i]+(h*k3vx))
            k4y = f2(vy[i]+(h*k3vy))
            k4vx = f3((x[i]+(h*k3x)),(y[i]+(h*k3y)))
            k4vy = f4((x[i]+(h*k3x)),(y[i]+(h*k3y)))
            
            x[i+1] = x[i] + (h/6)*(k1x + 2*k2x + 2*k3x + k4x)
            y[i+1] = y[i] + (h/6)*(k1y + 2*k2y + 2*k3y + k4y)
            vx[i+1] = vx[i] + (h/6)*(k1vx + 2*k2vx + 2*k3vx + k4vx)
            vy[i+1] = vy[i] + (h/6)*(k1vy + 2*k2vy + 2*k3vy + k4vy)
            
            i += 1
        
        if ((x[i])**2 + (y[i])**2)**0.5 < 6.67e6:
            print('Madness, you crashed into Earth')
        #this if statement indicates whether a crash has occured 
        
        fig, ax = plt.subplots()     
        earth = plt.Circle((0,0),6.67e6)
        ax.add_patch(earth)
        plt.plot(x,y)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.show()
            
        
    elif Myinput == 'b':
        print('You chose part',Myinput)
        Mm = 7.35e22 #mass of the moon
        Ym = 3.8e8 #distance between centre of earth and moon. Centre of earth is reference point.
        Rm = 1.74e6 #radius of the moon 
        
        def f1(vx):
            return (vx)

        def f2(vy):
            return (vy)
        
        def f3(x,y):
            return (-G*M*x/((x**2 + y**2)**(3/2)))-(Mm*G*x/((x**2 + (y-Ym)**2)**(3/2)))
        
        def f4(x,y):
            return (-G*M*y/((x**2 + y**2)**(3/2)))-(Mm*G*(y-Ym)/((x**2 + (y-Ym)**2)**(3/2)))
        
        t=np.linspace(t0,tmax,numpoints)
        dt=(tmax-t0)/(numpoints-1)
        x=np.zeros(numpoints)
        y=np.zeros(numpoints)
        vx=np.zeros(numpoints)
        vy=np.zeros(numpoints)
        t[0] = t0
        i=0
        h = dt
        
        Inputx = input('Enter a value for x (in meters greater than |6e6|): ')
        x0 = float(Inputx)
        x[0] = x0
        Inputy = input('Enter a value for y (in meters greater than |6e6|): ')
        y0 = float(Inputy)
        y[0] = y0
        Inputvx = input('Enter a value for vx (in m/s): ')
        vx0 = Inputvx
        vx[0] = vx0
        Inputvy = input('Enter a value for vy (in m/s): ')
        vy0 = Inputvy
        vy[0] = vy0
            
        while t[i] < tmax and ((x[i])**2 + (y[i])**2)**0.5 > 6.67e6 and ((x[i])**2 + (y[i]-Ym)**2)**0.5 > Rm:
            
            k1x = f1(vx[i])
            k1y = f2(vy[i])
            k1vx = f3(x[i],y[i])
            k1vy = f4(x[i],y[i])
            
            k2x = f1(vx[i]+(h*k1vx/2))
            k2y = f2(vy[i]+(h*k1vy/2))
            k2vx = f3((x[i]+(h*k1x/2)),(y[i]+(h*k1y/2)))
            k2vy = f4((x[i]+(h*k1x/3)),(y[i]+(h*k1y/2)))
            
            k3x = f1(vx[i]+(h*k2vx/2))
            k3y = f2(vy[i]+(h*k2vy/2))
            k3vx = f3((x[i]+(h*k2x/2)),(y[i]+(h*k2y/2)))
            k3vy = f4((x[i]+(h*k2x/2)),(y[i]+(h*k2y/2)))
            
            k4x = f1(vx[i]+(h*k3vx))
            k4y = f2(vy[i]+(h*k3vy))
            k4vx = f3((x[i]+(h*k3x)),(y[i]+(h*k3y)))
            k4vy = f4((x[i]+(h*k3x)),(y[i]+(h*k3y)))
            
            x[i+1] = x[i] + (h/6)*(k1x + 2*k2x + 2*k3x + k4x)
            y[i+1] = y[i] + (h/6)*(k1y + 2*k2y + 2*k3y + k4y)
            vx[i+1] = vx[i] + (h/6)*(k1vx + 2*k2vx + 2*k3vx + k4vx)
            vy[i+1] = vy[i] + (h/6)*(k1vy + 2*k2vy + 2*k3vy + k4vy)
            
            i += 1
        
        if ((x[i])**2 + (y[i])**2)**0.5 < 6.67e6:
            print('Madness, you crashed into Earth')
        elif ((x[i])**2 + (y[i]-Ym)**2)**0.5 < Rm:
            print('Madness, you crashed into the Moon')
        #building on from the part a, these if statements indicates whether you
        #have crashed into the earth or the moon 
        
        fig, ax = plt.subplots()     
        earth = plt.Circle((0,0),6.67e6)
        moon = plt.Circle((0,3.8e8),Rm)
        ax.add_patch(moon)
        ax.add_patch(earth)
        plt.plot(x,y)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.show()
    
    elif Myinput != 'q':
        print('Unavailable')
print('Quit')

'''
The code doing the integration is largely the same between parts 'a' and 'b'
however the defining functions before the integration changes depending on N-bodies
present in the system.
'''

'''
The integration using Runge-Kutta method, part of the code,
can be simplified using the scipy.integrate.solve_ivp, given that the functions
f1,f2, etc. are condensed into a single function. 
'''