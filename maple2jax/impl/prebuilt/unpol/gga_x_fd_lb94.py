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
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = t21 ** 2
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t26 = t22 / t24
t27 = math.sqrt(s0)
t29 = 2 ** (0.1e1 / 0.3e1)
t31 = 0.1e1 / t19 / r0
t36 = t26 * t27 * t29 * t31 / 0.12e2
t37 = fd_int0(t36)
t38 = math.log(t36)
t40 = fd_int1(t36)
t49 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 - t26 * t27 * t29 * t31 * (t37 * t38 - t40) / 0.12e2))
res = 0.2e1 * t49
