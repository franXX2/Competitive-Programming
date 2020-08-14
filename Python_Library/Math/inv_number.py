def xgcd(a, b):
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b != 0:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0

 """
    return x
    which satisfies
         : ax==1(mod m) in O(logM).
"""

def modinv(a, m):
    g, x, y = xgcd(a, m)
    if g != 1:
        return False
    else:
        return x % m

