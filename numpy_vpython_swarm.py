import vpython as vp
import numpy as np
import random
import scipy.spatial
import matplotlib.pyplot as plt

# create numball=10 spheres witth random positions and colors
numball=30
balls=[]
for i in range(numball):
    balls.append(vp.sphere(pos=vp.vector(random.random(),random.random(),random.random())-vp.vector(.5,.5,.5), color=vp.color.hsv_to_rgb(vp.vector(random.random(),1,1)), radius=0.1))

#give the balls random velocities
for ball in balls:
    ball.velocity=vp.vector(random.random()-.5,random.random()-.5,random.random()-.5)

#give the balls random masses
for ball in balls:
    ball.mass=random.random()

#first, create a list of the masses of all the balls
masslist=[]
for ball in balls:
    masslist.append(ball.mass)
#convert the list to a numpy array
massarray=np.array(masslist)



#set up the time step
dt=0.001

#set up the gravitational constant
J=.1

#draw a box around the scene of size boxsize x boxsize x boxsize
boxsize=5
box=vp.box(pos=vp.vector(0,0,0), length=boxsize, height=boxsize, width=boxsize, opacity=0.5,color=vp.color.white)

#set up the scene
vp.scene.width=800
vp.scene.height=800
vp.scene.range=boxsize
vp.scene.center=vp.vector(0.,0.,0.)
vp.scene.autoscale=False

#set up the rate at which the box shrinks
boxshrinkrate=0.1

#set up the time
t=0

#set up the arrays to store the data
kearray=[]
pearray=[]
tarray=[]

#set up an array to store the standard deviation of the momentum and energy of the system
momentumStDevArray=[]
energyStDevArray=[]

#set up the time loop
while boxsize>0.:
    #use numpy to calculate the distance between all pairs of balls
    #the distance between ball i and ball j is stored in the ith row and jth column of the matrix

    #first, create a list of the positions and velocities of all the balls
    poslist=[]
    vellist=[]    


    for ball in balls:
        poslist.append([ball.pos.x,ball.pos.y,ball.pos.z])
        vellist.append([ball.velocity.x,ball.velocity.y,ball.velocity.z])

    #find the magnitudes of the velocities of all the balls
    vellistmag=np.sqrt(np.sum(np.array(vellist)**2,axis=1))

    #calculate the standard deviation of the momentum
    momStDev=np.std(vellistmag*massarray)

    #append the standard deviation to the array
    momentumStDevArray.append(momStDev)

    #find the magnitude of the item in vellist with the largest magnitude
    maxvel=np.max(np.sqrt(np.sum(np.array(vellist)**2,axis=1)))

    #if the fastest ball is moving faster than the box is shrinking, decrease the time step
    # if maxvel>boxshrinkrate:
    #     dt=0.1*boxshrinkrate/maxvel
    # else:
    #     dt=0.001

    #convert the list to a numpy array
    posarray=np.array(poslist)
    velarray=np.array(vellist)
    #calculate the distance between all pairs of balls
    distancematrix=scipy.spatial.distance.cdist(posarray, posarray)

    #calculate the distance vector between all pairs of balls
    #the distance vector between ball i and ball j is stored in the ith row and jth column of the matrix
    distancevectormatrix=(posarray.T[...,np.newaxis]-posarray.T[:,np.newaxis]).T

    #calculate the unit vector between all pairs of balls
    #the unit vector between ball i and ball j is stored in the ith row and jth column of the matrix
    rhatmatrix=(distancevectormatrix.T / distancematrix).T
    rhatmatrix[np.isnan(rhatmatrix)] = 0 # replace NaN values with 0 for entries that were 0/0

    #calculate the mass matrix
    massmatrix=np.outer(massarray,massarray)

    #calculate the force between all pairs of balls
    #the force between ball i and ball j is stored in the ith row and jth column of the matrix
    forcematrix = J * massmatrix * distancematrix

    #calculate the acceleration of each ball
    #the acceleration of ball i is stored in the ith row of the matrix
    accmatrix = (rhatmatrix.T * forcematrix / massarray).T.sum(axis=1)

    #update the velocity array
    velarray=velarray+accmatrix*dt

    #update the position array
    posarray=posarray+velarray*dt

    #update the position of each ball
    for i in range(numball):
        balls[i].pos=vp.vector(posarray[i][0],posarray[i][1],posarray[i][2])
        balls[i].velocity=vp.vector(velarray[i][0],velarray[i][1],velarray[i][2])

    #constrain the balls to a box of size boxsize x boxsize x boxsize

    for ball in balls:
        if ball.pos.x>boxsize/2:
            # ball.pos.x=boxsize/2
            ball.velocity.x=-ball.velocity.x
        if ball.pos.x<-boxsize/2:
            # ball.pos.x=-boxsize/2
            ball.velocity.x=-ball.velocity.x
        if ball.pos.y>boxsize/2:
            # ball.pos.y=boxsize/2
            ball.velocity.y=-ball.velocity.y
        if ball.pos.y<-boxsize/2:
            # ball.pos.y=-boxsize/2
            ball.velocity.y=-ball.velocity.y
        if ball.pos.z>boxsize/2:
            # ball.pos.z=boxsize/2
            ball.velocity.z=-ball.velocity.z
        if ball.pos.z<-boxsize/2:
            # ball.pos.z=-boxsize/2
            ball.velocity.z=-ball.velocity.z
    #slowly shrink the box
    boxsize=boxsize-boxshrinkrate*dt
    box.length=boxsize
    box.height=boxsize
    box.width=boxsize

    #calculate the kinetic energy of each ball
    #the kinetic energy of ball i is stored in the ith row of the matrix
    keballmatrix=0.5*massarray*(vellistmag**2)
    #calculate the potential energy of each ball
    #the potential energy of ball i is stored in the ith row of the matrix
    peballmatrix=J*0.5*massmatrix*distancematrix**2
    peballmatrix=peballmatrix.sum(axis=1)/2

    #calculate the total energy of each ball
    #the total energy of ball i is stored in the ith row of the matrix
    teballmatrix=keballmatrix+peballmatrix

    #calculate the standard deviation of the total energy
    energyStDev=np.std(teballmatrix)

    #append the standard deviation to the array
    energyStDevArray.append(energyStDev)

    #calculate the total kinetic energy of the system
    ke=(0.5*massarray*(vellistmag**2)).sum()
    #calculate the total potential energy of the system
    pe=J*0.5*massmatrix*distancematrix**2
    pe=pe.sum()/2

    #store the total kinetic energy and total potential energy in an array
    kearray=np.append(kearray,ke)
    pearray=np.append(pearray,pe)
    #store the time in an array
    tarray=np.append(tarray,t)

    vp.rate(round(100/dt))

    #update the time
    t=t+dt

#plot the total kinetic energy and total potential energy as a function of time on a log scale
# plt.semilogy(tarray,kearray)
# plt.semilogy(tarray,pearray)
# plt.semilogy(tarray,kearray+pearray)
plt.plot(tarray,kearray)    
plt.plot(tarray,pearray)
plt.plot(tarray,kearray+pearray)

#label the axes
plt.xlabel('time')
plt.ylabel('energy')
#create a legend
plt.legend(['kinetic energy','potential energy','total energy'])
#give the plot a title with the number of balls and the value of the gravitational constant
plt.title('Energy of a system of '+str(numball)+' balls with J='+str(J))
#show the plot
plt.show()

#plot the standard deviation of the energy as a function of time
plt.plot(tarray,energyStDevArray)
#label the axes
plt.xlabel('time')
plt.ylabel('standard deviation of energy')
#give the plot a title with the number of balls and the value of the gravitational constant
plt.title('Standard deviation of energy of a system of '+str(numball)+' balls with J='+str(J))
#show the plot
plt.show()

