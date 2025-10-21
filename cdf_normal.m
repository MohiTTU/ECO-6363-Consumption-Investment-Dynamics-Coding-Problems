function c = cdf_normal(x)
    c = 0.5 * erfc(-x/sqrt(2));