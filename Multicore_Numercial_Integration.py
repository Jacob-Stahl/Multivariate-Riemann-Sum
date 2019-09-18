import numpy as np
import math
from  numba import jit, vectorize
import matplotlib.pyplot as plt

while True:

    input_func = str(input("input function of x and y : "))

    @vectorize(['float32(float32,float32)'],target = 'parallel')
    def zdA(z,dA):
        return z*dA

    def eval_function(a,b,c,d,num_dx,num_dy): # returns array of outputs within the specified limits
        xin = np.linspace(a, b, num_dx, endpoint=True)
        yin = np.linspace(c, d, num_dy, endpoint=True)

        z = np.zeros((num_dx,num_dy),dtype=np.float32)

        for i in range(num_dx):
            for j in range(num_dy):
                x = xin[i]
                y = yin[j]
                z[i][j] = eval(input_func)

        return z

    def surface_area(z,dx,dy):
        z1 = z[0:-1,0:-1]   #  z1  z2
        z2 = z[0:-1,1:  ]   #  z3  z4
        z3 = z[1:  ,0:-1]
        z4 = z[1:  ,1:  ]

        @vectorize(['float32(float32,float32,float32,float32,float32)'],target = 'parallel')
        def right_area(z1,z2,z4,dx,dy):
            return ( (((z1-z2)**2 + dx**2)**(1/2)) *
                     (((z4-z2)**2 + dy**2)**(1/2)) / 2 )

        @vectorize(['float32(float32,float32,float32,float32,float32)'],target = 'parallel')
        def left_area(z1,z3,z4,dx,dy):
            return ( (((z1-z3)**2 + dy**2)**(1/2)) *
                     (((z3-z4)**2 + dx**2)**(1/2)) / 2 )

        return np.sum(left_area(z1,z3,z4,dx,dy)) + np.sum(right_area(z1,z2,z4,dx,dy))

    limit_a = float(input('limit a: '))
    limit_b = float(input('limit b: '))
    limit_c = float(input('limit c: '))
    limit_d = float(input('limit d: '))

    num_dx = int(500)
    num_dy = int(500)

    dx      = (limit_b - limit_a) / (num_dx) # differential for volume
    dy      = (limit_d - limit_c) / (num_dy)
    surf_dx = (limit_b - limit_a) / (num_dx -1) # differential for surface area
    surf_dy = (limit_d - limit_c) / (num_dy -1)
    
    dA = dx*dy # differential area

    z = eval_function(limit_a,limit_b,limit_c,limit_d,num_dx,num_dy)

    volume   = np.sum(zdA(z,dA))
    surf_a   = surface_area(z,surf_dx,surf_dy)

    print('volume: ',volume,'   surface area:  ',surf_a)

    x = np.linspace(limit_c, limit_d, num_dy)
    y = np.linspace(limit_a, limit_b, num_dx)

    plt.imshow(z, cmap='jet', interpolation='nearest')
    plt.show()
    print()
