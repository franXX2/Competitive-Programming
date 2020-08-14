#フェルマーテスト
#ある自然数Nが与えられた時に、それが素数であるかどうかをO(logN)で返す。
#それが正しい確率は大体1-10**(-9)である。(ほぼ正確)
import numpy as np
from math import gcd

def fast_is_prime(p):
    for _ in range(100):
        a = np.random.choice(p)
        if gcd(a, p) > 1 or pow(a, p-1, p) != 1:
            return False
    return True

    
