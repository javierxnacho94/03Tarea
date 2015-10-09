#Tarea 3
#Javier Silva Lafaurie
#RUT: 18.928.657-0
#Pregunta 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

mu=1.657 #definimos el mu segun el rut

def funcion(y,v,mu=1.657): #funcion de la ecuacion diferencial
    return v, -y-mu*((y**2) -1)*v

def K1(y,v,h): #funcion para calcular el K1
    var=funcion(y,v)
    return h*var[0], h*var[1]

def K2(y,v,h): #funcion para calcular el K2
    k1=K1(y,v,h)
    var=funcion(y+k1[0]/2, v+k1[1]/2)
    return h*var[0], h*var[1]

def K3(y,v,h): #funcion para calcular el K3
    k1=K1(y,v,h)
    k2=K2(y,v,h)
    var=funcion(y-k1[0]-2*k2[0], v-k1[1]-2*k2[1])
    return h*var[0], h*var[1]

def RK3(s0,y0,v0,sf): #funcion para resolver la edo numericamente con range-kuta orden 3
    s=np.linspace(s0,sf,500) #vector discreto equispaciado de los s con 500 valores
    h=s[1]-s[0] #intervalo h
    y=np.ones(len(s)) #vectores de y y v llenos de 1s
    v=np.ones(len(s))
    y[0]=y0 #guardamos condiciones iniciales
    v[0]=v0

    for i in range(0,len(s)-1): #iteramos para calcular y_i+1 y v_i+1 dados v_i y y_i
        k1=K1(y[i],v[i],h)
        k2=K2(y[i],v[i],h)
        k3=K3(y[i],v[i],h)

        y[i+1]=y[i]+(k1[0]+4*k2[0]+k3[0])/6
        v[i+1]=v[i]+(k1[1]+4*k2[1]+k3[1])/6

    return s,y,v #devolvemos el vector s y los resultados correspondientes de y y v

espacio=RK3(0,0.1,0,20*np.pi) #Damos valores a la funcion de condiciones iniciales y valor final de s, que es 20pi

plt.plot(espacio[0],espacio[1]) #ploteamos trayectoria
plt.xlabel('s')
plt.ylabel('y')
plt.title('Trayectoria y(s) para el oscilador de van der Pool')
plt.savefig('trayectoria.png')
#plt.show()
plt.figure()
plt.plot(espacio[1],espacio[2]) #ploteamos espacio de fase
plt.xlabel('y')
plt.ylabel('dy/ds')
plt.title('Trayectoria en el espacio de fase (y,dy/ds) para el oscilador de van der Pool')
plt.savefig('espacio_fase.png')
#plt.show()

#Pregunta 2

sigma=10. #guardamos los parametros dados
beta=8/3.
rho=28.

def sistema(t,w,sigma=sigma,beta=beta,rho=rho): #definimos el sistema de ecuaciones de la EDO
    x, y, z =w
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

t0=0 #damos condiciones iniciales
x0=10.
y0=20.
z0=30.
w0= x0, y0, z0
tf=40 #tiempo final de integracion

r=ode(sistema) #usamos ode para resolver la EDO
r.set_integrator('dopri5') #con 'dopri5' se resuelve con el metodo de range-kuta 4
r.set_initial_value(w0) #guardamos los valores iniciales

t=np.linspace(t0,tf,5000) #creamos un vector de tiempos de integracion con una precision de 5000 valores

x=np.ones(len(t)) #creamos vectores para ir guardando la integracion
y=np.ones(len(t))
z=np.ones(len(t))

for i in range(0,len(t)): #se inicia la integracion
    r.integrate(t[i])
    x[i], y[i], z[i] = r.y

fig=plt.figure()
fig.clf()
ax = fig.add_subplot(111, projection='3d') #se prepara para plotear en 3D
ax.set_aspect('equal')
ax.plot(x, y, z)          #Se plotean los valores calculados con ode
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.title('Atractor de Lorentz')
plt.savefig('Atractor_lorentz.png')

plt.show()
