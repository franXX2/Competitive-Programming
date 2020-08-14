#import numpy as np
def find_sequence(N,Terms,coefficients,mod):
    
    #given the sequence in the form below:
        #[first K term] := a0,a1.......a(k-1)
        #[recurrence relation] := a(n+k)= b(k-1)*a(n+k-1)+....... b(0)*a(n)
        
    #Then,define Terms and coefficients as follows:
        #Terms := [a0....... ak]
        #coefficients :=[b0.....bk]
        
    # return a_n  %mod  in O(K**2 log N), which is much faster than matrix_binary_powering_method,which works in O(K**3 logN).
    #Note that mod*K<2**64 in order to avoid overflow error.
    
    assert len(Terms)==len(coefficients)
    K=len(coefficients)
    data=[N]
    while N:
        if N%2==0:
            N=N//2
            data.append(N)
        else:
            N-=1
            data.append(N)
    data.reverse()
    
    C = np.array([0]*K)
    old_C = np.array([0]*K)
    tmp_C=np.array([0]*K)
    Cs = np.array([[0]*K for _ in range(K)])
    cofs=np.array(coefficients)
    #C(0,i) の定義
    C[0] = 1
    for i in range(len(data)-1):
        now, nex = data[i], data[i+1]
        old_C*=0
        old_C += C
        if nex == 1+now:
            C=old_C[K-1]*cofs
            C%=mod
            for i in range(1,K):
                C[i]+=old_C[i-1]
            C%=mod
            continue
        else:
            for i in range(K):
                Cs[i]*=0
            Cs[0]+=C
            for i in range(1,K):
                Cs[i]=Cs[i-1][K-1]*cofs
                Cs[i]%=mod
                for j in range(1,K):
                    Cs[i][j]+=Cs[i-1][j-1]
                Cs[i]%=mod
            C*=0
            Cs=Cs.T
            for i in range(K):
                tmp_C=0
                tmp_C=old_C*Cs[i]
                tmp_C%=mod
                C[i]=np.sum(tmp_C)
                C[i]%=mod
    ans=0
    for i in range(K):
        ans+=Terms[i]*C[i]
    ans%=mod
    return ans

def matrix_power(A, N, mod):
    # returnA^N %moｄ in O(K**3 log N). (K is the size of A.)
    assert A.shape[0] == A.shape[1]
    K = A.shape[0]
    if N == 0:
        return np.eye(K, dtype=np.int64)
    else:
        if N % 2 == 0:
            mat = matrix_power(A, N//2, mod)
            return np.dot(mat, mat) % mod
        else:
            mat = matrix_power(A, N//2, mod)
            return np.dot(np.dot(mat, mat) % mod, A) % mod


def Fibonacci(N, mod):
    # return the n-th term of the fivonacci sequence  in O(logN).
    # F0=0,F1=1
    d = np.array([1, 0])
    A = np.array([[1, 1], [1, 0]], dtype=np.int64)
    res = np.dot(matrix_power(A, N, mod), d)
    return int(res[-1]) % mod


def fibonacci(N, mod):
    # returns the n-th term %mod of fibbonacci seuqnce  in O(logN).
    # you don't need to care about overflow.
    f1, f = 1, 1
    r1, r = 1, 0
    while N:
        if N & 1:
            r1, r = (f1*r1+f*r) % mod, (f1*r+f*(r1-r)) % mod
        f1, f = (f1**2+f**2) % mod, f*(2*f1-f) % mod
        N >>= 1
    return r


def find_fibonacci_cycle(P):
    # when modular is a prime number, returns fibonacci_periodic_cycle in O(1).
    if P % 5 == 1 or P % 5 == 4:
        return P-1
    elif P % 5 == 2 or P % 5 == 3:
        return 2*P+2
    else:
        return 20

class Combi():
    def __init__(self, N, mod):
        self.power = [1 for _ in range(N+1)]
        self.rev = [1 for _ in range(N+1)]
        self.mod = mod
        for i in range(2, N+1):
            self.power[i] = (self.power[i-1]*i) % self.mod
        self.rev[N] = pow(self.power[N], self.mod-2, self.mod)
        for j in range(N, 0, -1):
            self.rev[j-1] = (self.rev[j]*j) % self.mod

    def C(self, K, R):
        if K < R:
            return 0
        else:
            return ((self.power[K])*(self.rev[K-R])*(self.rev[R])) % self.mod

    def P(self, K, R):
        if K < R:
            return 0
        else:
            return (self.power[K])*(self.rev[K-R]) % self.mod
    

def chinese_reminder(pair1, pair2):
    """
    Let pair 1 be (x1,y1) and  pair 2 be (x2,y2).
    Now returns N: which satifsfies

        N == x1 (mod y1)
        N == x2 (mod y2)

    in the form of tuple (P,Q) [when N 　N==P (mod Q) ] in  O(logtime).

    if there's no such a number, returns
    (float(inf),float(inf)) """

    x1, y1 = pair1
    x2, y2 = pair2
    g = math.gcd(y1, y2)
    if (x2-x1) % g != 0:
        return (float("inf"), float("inf"))
    else:
        K = (x2-x1)//g
        y1, y2 = y1//g, y2//g
        t = -K*modinv(y2, y1)
        m = x2+t*g*y2
        return (m % (g*y1*y2), g*y1*y2)