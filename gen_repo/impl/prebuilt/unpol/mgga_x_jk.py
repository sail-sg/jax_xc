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
t21 = t3 ** 2
t24 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t26 = 4 ** (0.1e1 / 0.3e1)
t29 = 2 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t31 = s0 * t30
t32 = r0 ** 2
t33 = t19 ** 2
t34 = t33 * t32
t35 = 0.1e1 / t34
t37 = math.sqrt(s0)
t40 = 0.1e1 / t19 / r0
t44 = math.asinh(t37 * t29 * t40)
t70 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.9e1 * params_a_beta * t21 / t24 * t26 * t31 * t35 / (params_a_gamma * params_a_beta * t37 * t29 * t40 * t44 + 0.1e1) / (0.1e1 + (t31 * t35 - l0 * t30 / t33 / r0) / s0 * t29 * t34)))
res = 0.2e1 * t70
