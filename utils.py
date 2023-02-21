from decimal import Decimal


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def exponential_decay(x, a, c=100):
    #\left(\frac{\left(b-b^{\left(x+1\right)}\right)}{b-1}+b^{x}\right)
    #simplified: \frac{-b^{x}+b}{b-1}
    #1>b>0
    # The higher the value of c, the more the decay is delayed
    x = Decimal(x)
    a = Decimal(a)
    c = Decimal(c)
    b = a ** c
    res = ((-b**x)+b)/(b - 1)
    return float(res)


#def exponential_decay(x, b=1E-6, a=0.01, n=1):
#    #b^{\frac{x}{an}}
#    return b**(x/(a*n))


#def exponential_decay(x, b=1E-6, a=2):
#    #b^{\frac{x}{2}}
#    return b**(x/a)