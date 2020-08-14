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

    
import random

#ミラービンの確率的素数判定法.(10**10程度の自然数に対してはM=40程度で十分正確かつ高速に判定を行うことができる。)
def Is_Prime(X, M):
    if X == 1:
        return False
    if X == 2:
        return True
    K = X-1
    order = 0
    while K % 2 == 0:
        order += 1
        K = K//2
    for i in range(M):
        if trial(X, order, K) == False:
            return False
    return True


def trial(X, order, K):
    num = random.randint(2, X-1)
    b = pow(num, K, X)
    if b == 1:
        return True
    for j in range(order):
        if (b+1) % X == 0:
            return True
        b = b**2 % X
    return False