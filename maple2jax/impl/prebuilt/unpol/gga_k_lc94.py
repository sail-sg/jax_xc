t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = jnp.where(t8, -t9, 0)
t12 = jnp.where(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = jnp.where(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t25 = 6 ** (0.1e1 / 0.3e1)
t27 = math.pi ** 2
t28 = t27 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t30 = 0.1e1 / t29
t32 = 2 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t35 = r0 ** 2
t38 = s0 * t33 / t23 / t35
t41 = math.exp(-params_a_alpha * t25 * t30 * t38 / 0.24e2)
t48 = t25 ** 2
t49 = 0.1e1 / t28
t50 = t48 * t49
t51 = math.sqrt(s0)
t54 = 0.1e1 / t22 / r0
t55 = t51 * t32 * t54
t58 = (t50 * t55 / 0.12e2) ** params_a_expo
t59 = params_a_f * t58
t67 = math.asinh(params_a_b * t48 * t49 * t55 / 0.12e2)
t79 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 + ((params_a_d * t41 + params_a_c) * t25 * t30 * t38 / 0.24e2 - t59) / (0.1e1 + t50 * t51 * t32 * t54 * params_a_a * t67 / 0.12e2 + t59)))
res = 0.2e1 * t79
