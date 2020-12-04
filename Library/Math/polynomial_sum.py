def poly_sum(X, N, MOD):
    """ caliculating X^N +    .... X^1 + 1 modulo MOD in logarithmic times.
    """
    if N == 0:
        return 1
    if N == 1:
        return (X+1) % MOD
    else:
        if N % 2 == 0:
            tmp_res = poly_sum(X, N//2, MOD)
            return ((pow(X, N//2, MOD)+1)*tmp_res-pow(X, N//2, MOD)) % MOD
        elif N % 2 != 0:
            tmp_res = poly_sum(X, N//2, MOD)
            return (pow(X, N//2+1, MOD)+1)*tmp_res % MOD

