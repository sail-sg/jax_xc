t1 = r0 ** 2
t2 = 0.1e1 / t1
t6 = 0.2e1 * tau0 * t2
t10 = s0 / t1 / r0 / 0.4e1
t12 = 0.1e1 / math.pi
t13 = (l0 * t2 / 0.2e1 - t6 + t10) * t12
t15 = lax_cond(-0.9999999999e0 < t13, t13, -0.9999999999e0)
t16 = math.exp(-1)
t18 = scipy.special.lambertw(t15 * t16)
t21 = scipy.special.i0(t18 / 0.2e1 + 0.1e1 / 0.2e1)
t23 = t6 - t10
t25 = lax_cond(0.1e-9 < t23, t23, 0.1e-9)
t26 = math.sqrt(t25)
t30 = math.sqrt(0.2e1)
t32 = math.sqrt(r0)
res = -(math.pi * t21 - 0.4e1 / 0.3e1 * t12 * t26) * t30 * t32 / 0.2e1
