class SegTree():
    """
        This Tree Data Structure is suitable for managing informations of segments.
        Any Data Type which has Monoid Structre can be applied.
        All queries finish in finish in logarithmic times.
    """
    def __init__(self, N, function=min, basement=(1 << 60)):
        self.n = N
        self.K = (self.n-1).bit_length()
        self.f = function
        self.b = basement
        self.seg = [basement]*(1 << (self.K+1)+1)
 
    def all_update(self, LIST):
        # a[i] -> LIST[i] for all i
        assert len(LIST) < self.n
        X = 1 << (self.K)
        for i, v in enumerate(LIST):
            self.seg[i + X] = v
        for i in range(X - 1, 0, - 1):
            self.seg[i] = self.f(self.seg(i << 1), self.seg[i << 1 | 1])
 
    def update(self, k, value):
        # a[i] -> value
        X = 1 << (self.K)
        k += X
        self.seg[k] = value
        while k:
            k = k >> 1
            self.seg[k] = self.f(self.seg[k << 1], self.seg[(k << 1) | 1])
 
    def get_value(self, I):
        # a[i]
        return self.seg[I + (1 << (self.K))]
 
    def query(self, L, R):
        # f [L,R)
        num = 1 << (self.K)
        L += num
        R += num
        vL = self.b
        vR = self.b
        while L < R:
            if L & 1:
                vL = self.f(vL, self.seg[L])
                L += 1
            if R & 1:
                R -= 1
                vR = self.f(self.seg[R], vR)
            L >>= 1
            R >>= 1
        return self.f(vL, vR)
 
    def find_max_index(self, L, R, X):
        # maximum of i which satisfies:
            ## i is in [L,R)
            ## a[i] <= X 
        # -1 if not exists.

        return self.fMi(L, R, X, 1, 0, 1<<(self.K))

    def find_min_index(self, L, R, X):
        # minimum of i which satisfies:
            ## i is in [L,R)
            ## a[i] <= X 
        # -1 if not exists.
        return self.fmi(L, R, X, 1, 0, 1<<(self.K))

    def fMi(self, a, b, x, k, l, r):
        if self.seg[k] > x or r <= a or b <= l:
            # a[i] >=X -> change from "self.seg[k] > x" to "self.seg[k] <x"
            return -1
        else:
            if k >= 1<<(self.K):
                return k-(1<<self.K)
            else:
                vr = self.fMi(a, b, x, (k << 1) | 1, (l + r) // 2, r)
                if vr != -1:
                    return vr
                return self.fMi(a, b, x, k << 1, l, (l + r) // 2)

    def fmi(self, a, b, x, k, l, r):
        if self.seg[k] > x or r <= a or b <= l:
            # a[i] >=X -> change from "self.seg[k] > x" to "self.seg[k] <x"
            return -1
        else:
            if k >= (1<<self.K):
                return k-(1<<self.K)
            else:
                vl = self.fmi(a, b, x, k << 1, l, (l+r)//2)
                if vl != -1:
                    return vl
                return self.fmi(a, b, x, k << 1 | 1, (l+r)//2, r)
