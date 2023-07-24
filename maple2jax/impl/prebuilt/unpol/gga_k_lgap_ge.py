t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = lax_cond(t8, -t9, 0)
t12 = lax_cond(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = lax_cond(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t26 = 6 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = math.pi ** 2
t30 = t29 ** (0.1e1 / 0.3e1)
t33 = math.sqrt(s0)
t34 = 2 ** (0.1e1 / 0.3e1)
t43 = t30 ** 2
t46 = t34 ** 2
t48 = r0 ** 2
t58 = t48 ** 2
t67 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 + params_a_mu[0] * t27 / t30 * t33 * t34 / t22 / r0 / 0.12e2 + params_a_mu[1] * t26 / t43 * s0 * t46 / t23 / t48 / 0.24e2 + params_a_mu[2] / t29 * t33 * s0 / t58 / 0.24e2))
res = 0.2e1 * t67
