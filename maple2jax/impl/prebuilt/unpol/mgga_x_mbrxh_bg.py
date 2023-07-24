t3 = 0.1e1 <= p_a_zeta_threshold
t4 = p_a_zeta_threshold - 0.1e1
t6 = lax_cond(t3, -t4, 0)
t7 = lax_cond(t3, t4, t6)
t8 = 0.1e1 + t7
t10 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t12 = t8 ** (0.1e1 / 0.3e1)
t14 = lax_cond(t8 <= p_a_zeta_threshold, t10 * p_a_zeta_threshold, t12 * t8)
t15 = r0 ** (0.1e1 / 0.3e1)
t18 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t21 = 4 ** (0.1e1 / 0.3e1)
t22 = 2 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t25 = t15 ** 2
t30 = 6 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t32 = math.pi ** 2
t33 = t32 ** (0.1e1 / 0.3e1)
t34 = t33 ** 2
t38 = r0 ** 2
t43 = s0 ** 2
t45 = t38 ** 2
t51 = 0.46864e0 * tau0 * t23 / t25 / r0 - 0.3e1 / 0.10e2 * t31 * t34 + 0.89e-1 * s0 * t23 / t25 / t38 + 0.106e-1 * t43 * t22 / t15 / t45 / r0
t52 = abs(t51)
t55 = lax_cond(0.0e0 < t51, 0.50e-12, -0.50e-12)
t56 = lax_cond(t52 < 0.50e-12, t55, t51)
t57 = br89_x(t56)
t59 = math.exp(t57 / 0.3e1)
t61 = math.exp(-t57)
t71 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -t14 * t15 / t18 * t21 * t59 * (0.1e1 - t61 * (0.1e1 + t57 / 0.2e1)) / t57 / 0.4e1)
res = 0.2e1 * t71
