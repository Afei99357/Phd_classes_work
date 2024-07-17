import math

n = 10
i = 0
p = 0.1029
x = []
for i in range(10):
    i_factory = math.factorial(i)
    n_factory = math.factorial(n)
    n_i_difference_factory = math.factorial(n-i)
    x.append(n_i_difference_factory / (i_factory * n_i_difference_factory) * math.pow(p, i) * math.pow(1-p, n-i))
    i = i + 1


print(x)