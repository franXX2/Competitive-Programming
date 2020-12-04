
class Combi():
    """
    This Data Structure helps you to caliculate Binomial coefficient: nCr (n Chooses r) in modulo MOD.
    Note that this only works fast enough when n is not so big.(aploximately n < 1e7 or so).
    """
    def __init__(self, N, mod=998244353):
        self.power = [1 for _ in range(N+1)]
        self.rev = [1 for _ in range(N+1)]
        self.mod = mod
        for i in range(2, N+1):
            self.power[i] = (self.power[i-1]*i) % self.mod
        self.rev[N] = pow(self.power[N], self.mod-2, self.mod)
        for j in range(N, 0, -1):
            self.rev[j-1] = (self.rev[j]*j) % self.mod

    def com(self, K, R):
        if not (0 <= K <= R):
            return 0
        else:
            return ((self.power[K])*(self.rev[K-R])*(self.rev[R])) % self.mod

    def perm(self, K, R):
        if not (0 <= K <= R):
            return 0
        else:
            return (self.power[K])*(self.rev[K-R]) % self.mod
