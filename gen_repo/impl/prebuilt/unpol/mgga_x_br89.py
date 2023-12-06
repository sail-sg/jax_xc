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
t20 = 4 ** (0.1e1 / 0.3e1)
t23 = 2 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t26 = t15 ** 2
t28 = 0.1e1 / t26 / r0
t36 = r0 ** 2
t42 = l0 * t24 * t28 / 0.6e1 - 0.2e1 / 0.3e1 * params_a_gamma * tau0 * t24 * t28 + params_a_gamma * s0 * t24 / t26 / t36 / 0.12e2
t43 = abs(t42)
t46 = lax_cond(0.0e0 < t42, 0.50e-12, -0.50e-12)
t47 = lax_cond(t43 < 0.50e-12, t46, t42)
t48 = br89_x(t47)
t50 = math.exp(t48 / 0.3e1)
t51 = math.exp(-t48)
t58 = 6 ** (0.1e1 / 0.3e1)
t59 = t58 ** 2
t60 = math.pi ** 2
t61 = t60 ** (0.1e1 / 0.3e1)
t62 = t61 ** 2
t64 = 0.3e1 / 0.10e2 * t59 * t62
t66 = tau0 * t24 * t28
t67 = t64 - t66
t68 = t64 + t66
t71 = t67 ** 2
t73 = t68 ** 2
t78 = t71 ** 2
t80 = t73 ** 2
t91 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -t14 * t15 / t18 * t20 * t50 * (0.1e1 - t51 * (0.1e1 + t48 / 0.2e1)) / t48 * (0.1e1 + params_a_at * (t67 / t68 - 0.2e1 * t71 * t67 / t73 / t68 + t78 * t67 / t80 / t68)) / 0.4e1)
res = 0.2e1 * t91
