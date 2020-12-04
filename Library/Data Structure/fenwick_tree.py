class fenwick():
    """
    This Tree Data Structure speeds up caliculating summations of partial sum 
    and also updating subsets of sequences. Both queries finish in logarithmic times.
    """
    # 1-indexed
    def __init__(self, n):
        self.n = n
        self.data = [0]*(n+1)

    def to_sum(self, i):
        # return sigma(a_j) (0<=j<=i)
        s = 0
        while i > 0:
            s += self.data[i]
            i -= (i & -i)
        return s

    def add(self, i, x):
        #a_i -> a_i + x
        while i <= self.n:
            self.data[i] += x
            i += (i & -i)

    def get(self, i, j):
        # return sigma(a_k) (i<=k<=j)
        # assert 1<=i<=j<= N
        return self.to_sum(j)-self.to_sum(i-1)
