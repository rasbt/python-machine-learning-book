import theano
from theano import tensor as T
import numpy as np

# quadratic polynomial root example
# ax^2 + bx + c = 0
a = T.scalar('a')
b = T.scalar('b')
c = T.scalar('c')

core = b*b - 4*a*c
root_p = (-b + np.sqrt(core))/(2*a)
root_m = (-b - np.sqrt(core))/(2*a)

# compile
f = theano.function(inputs = [a, b, c], outputs = [root_p, root_m])

# run
polys = [[1, 2, 1],
         [1, -7, 12],
         [1, 0, 1]
        ]

for poly in polys:
    a, b, c = poly
    root1, root2 = f(a, b, c)
    print(root1, root2)
