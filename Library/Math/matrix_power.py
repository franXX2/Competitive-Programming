def matrix_power(A, N, MOD):
    """ caliculating A^N modulo MOD in O(K**3 log N). (K is the size of A.)
    """
    K = len(A)
    assert len(A[0]) == K

    def mat_mul(X, Y):
        assert len(X[0]) == len(Y)
        i, j = len(X), len(X[0])
        j, k = len(Y), len(Y[0])
        res = [[0]*k for _ in range(i)]
        for _ in range(i):
            for __ in range(k):
                res[_][__] = sum(X[_][itr]*X[itr][__]
                                 for itr in range(j)) % MOD
        return res
    if N == 0:
        res = [[0]*K for i in range(K)]
        for i in range(K):
            res[i][i] = 1
        return res
    else:
        if N % 2 == 0:
            mat = matrix_power(A, N//2, MOD)
            return mat_mul(mat, mat)
        else:
            mat = matrix_power(A, N//2, MOD)
            return mat_mul(mat_mul(mat, mat), A)
