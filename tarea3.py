#Tarea 3
#Javier Silva Lafaurie
#RUT: 18.928.657-0
#Pregunta 1

import numpy as np
import matplotlib.pyplot as plt

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
plt.title('Trayectoria y(s)')
plt.savefig('trayectoria.png')
#plt.show()
plt.figure()
plt.plot(espacio[1],espacio[2]) #ploteamos espacio de fase 
plt.xlabel('y')
plt.ylabel('dy/ds')
plt.title('Trayectoria en el espacio de fase (y,dy/ds)')
plt.savefig('espacio_fase.png')
plt.show()
