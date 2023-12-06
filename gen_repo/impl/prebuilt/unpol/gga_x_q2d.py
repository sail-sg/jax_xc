t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t20 ** 2
t35 = t21 / t24 * s0 * t28 / t31 / t30
t41 = t21 ** 2
t45 = s0 ** 2
t47 = t30 ** 2
t58 = math.sqrt(s0)
t64 = (t41 / t23 * t58 * t27 / t20 / r0) ** 0.35e1
t71 = t22 ** 2
t75 = t47 ** 2
t84 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * ((0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t35)) * (0.100e3 - t41 / t23 / t22 * t45 * t27 / t20 / t47 / r0 / 0.288e3) + 0.87153829697982569831e-4 * t64 * (0.1e1 + t35 / 0.24e2)) / (0.100e3 + 0.1e1 / t71 * t45 * s0 / t75 / 0.576e3))
res = 0.2e1 * t84
