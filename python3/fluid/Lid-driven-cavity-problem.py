#Lid Driven Cavity flow
#Steady State navier stokes equations are solved by five point finite point method
#\psi_{xx}+\psi_{yy}=-\omega
#\omega_{xx}+\omega_{yy}=U\psi_{x}+V\psi_{y}

from numpy import *
from matplotlib.pyplot import *

nx=50                 # Increase grid size as per requirment
ny=nx                  # Square meshing
h=1.0/nx               # Uniform step size
#dt=0.02
Re=1.0e1               # Change Re value, (tested upto 1000 )

maxerr=1.0e-5
SUR=0.8                # 0< SUR value <1,  Reduce it if fails to converge

sf=1.0e-15*ones((nx+1,ny+1))
vt=1.0e-15*ones((nx+1,ny+1))

sf1=zeros((nx+1,ny+1))
vt1=zeros((nx+1,ny+1))

U=zeros((nx+1,ny+1))
V=zeros((nx+1,ny+1))

plx=linspace(0,1,nx+1)       #For Ploting
ply=linspace(0,1,ny+1)       #For plotting
X,Y= meshgrid(plx,ply)


ite=0
err=maxerr+1.0

for j in range(ny+1):                       # Boundary values for stream function
    sf[j][ny]=0.0                           #Stream function is zero on the boundary
    sf[nx][j]=0.0
    sf[0][j]=0.0
    sf[j][0]=0.0
    
    U[j][ny]=1.0                           #U velocity
    U[nx][j]=0.0
    U[0][j]=0.0
    U[j][0]=0.0
    
    V[j][ny]=0.0                           #V velocity
    V[nx][j]=0.0
    V[0][j]=0.0
    V[j][0]=0.0


while(err>maxerr):
    ite=ite+1
    
    #Update old values
    
    for i in range(nx+1):
        for j in range(ny+1):
            sf1[i][j]=sf[i][j]
            vt1[i][j]=vt[i][j]

    #Solve for stream function

    for i in range(1,nx,1):
        for j in range(1,ny,1):
            sum1=0.25*(sf[i+1][j]+sf[i-1][j]+sf[i][j+1]+sf[i][j-1]+(h**2)*vt[i][j])
            sf[i][j]=(1.0-SUR)*sf[i][j]+SUR*sum1
        
    
    #Calculate U and V
    for i in range(1,nx,1):
        for j in range(1,ny,1):
            U[i][j]=(sf[i][j+1]-sf[i][j-1])/(2.0*h)
            V[i][j]=(sf[i-1][j]-sf[i+1][j])/(2.0*h)

    #Vorticity at the boundary
    #corner point singularities neglected

    #At bottom
    for i in range(1,nx,1):
        vt[i][0]=-2.0*sf[i][1]/(h*h)
    #At top
    for i in range(0,nx+1,1):
        vt[i][ny]=-2.0*sf[i][ny-1]/(h*h)-2.0/h
    #At left
    for j in range(1,ny,1):
        vt[0][j]=-2.0*sf[1][j]/(h*h)
    #At right
    for j in range(0,ny+1,1):
        vt[nx][j]=-2.0*sf[nx-1][j]/(h*h)
        
    for i in range(1,nx,1):
        for j in range(1,ny,1):
            sum1=(0.25-0.125*h*Re*U[i][j])*vt[i+1][j]+(0.25+0.125*h*Re*U[i][j])*vt[i-1][j]+(0.25-0.125*h*Re*V[i][j])*vt[i][j+1]+(0.25+0.125*h*Re*V[i][j])*vt[i][j-1]
            vt[i][j]=(1.0-SUR)*vt[i][j]+SUR*sum1
            
    err=amax(absolute([absolute(vt-vt1),absolute(sf-sf1)]))
    print('Iteration ',ite, "  ",err)
    
    
    #uncomment for live animation 
    '''
    clf()
    CS = contour(transpose(sf))
    clabel(CS, inline=1, fontsize=7)
    pause(0.001)
    '''   
CS = contour(X,Y,transpose(sf),20)
clabel(CS, inline=1, fontsize=10)
show()

CV = contour(X,Y,transpose(vt),20)
clabel(CV, inline=1, fontsize=10)
show()