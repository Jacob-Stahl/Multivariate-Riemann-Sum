import numpy as np
from  numba import jit, vectorize
import matplotlib.pyplot as plt

while True:

    @vectorize(['float32(float32,float32)'],target = 'parallel')
    def func(x,y): # returns outputs from x and y arrays
        return np.log(x*y) # function of x and y to be integrated

    @vectorize(['float32(float32,float32)'],target = 'parallel')
    def zdA(z,dA):
        return z*dA #multiplies z values by differential areas

    @jit
    def gen_inputs(a,b,c,d,num_dx,num_dy): # returns array of x inputs and array of y inputs to be plugged into "func(x,y)"
        x = np.linspace(a, b, num_dx, endpoint=True)
        y = np.linspace(c, d, num_dy, endpoint=True)

        out_x = np.zeros((num_dx,num_dy),dtype=np.float32)
        out_y = np.zeros((num_dx,num_dy),dtype=np.float32)

        for i in range(num_dx):
            out_x[:][i] = x
        for i in range(num_dy):
            out_y[i][:] = y
        out_y = np.transpose(out_y)

        print(out_y , '  out_y')
        print()
        print(out_x , '  out_x')
        print()

        return out_x , out_y

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

    num_dx = int(5000)
    num_dy = int(5000)

    dx      = (limit_b - limit_a) / (num_dx)
    dy      = (limit_d - limit_c) / (num_dy)
    surf_dx = (limit_b - limit_a) / (num_dx -1)
    surf_dy = (limit_d - limit_c) / (num_dy -1)
    
    dA = dx*dy

    x, y = gen_inputs(limit_a,limit_b,limit_c,limit_d,num_dx,num_dy)
    z = func(x,y)

    volume   = np.sum(zdA(z,dA))
    surf_a   = surface_area(z,surf_dx,surf_dy)

    print('volume: ',volume,'   surface area:  ',surf_a)

    x = np.linspace(limit_c, limit_d, num_dy)
    y = np.linspace(limit_a, limit_b, num_dx)

    plt.imshow(z, cmap='jet', interpolation='nearest')
    plt.show()
    print()
