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
t21 = t3 ** 2
t24 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t25 = 0.1e1 / t24
t26 = 4 ** (0.1e1 / 0.3e1)
t30 = 2 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = r0 ** 2
t34 = t20 ** 2
t37 = math.sqrt(s0)
t38 = t37 * t30
t40 = 0.1e1 / t20 / r0
t42 = math.asinh(t38 * t40)
t52 = 0.1e1 + 0.93333333333333333332e-3 * t21 * t25 * t26 * s0 * t31 / t34 / t33 / (0.1e1 + 0.2520e-1 * t38 * t40 * t42)
t56 = math.sqrt(math.pi * t21 * t25 * t26 / t52)
t60 = (t12 * r0) ** (0.1e1 / 0.3e1)
t64 = p_a_cam_omega / t56 * t30 / t60 / 0.2e1
t66 = 0.135e1 < t64
t67 = lax_cond(t66, t64, 0.135e1)
t68 = t67 ** 2
t71 = t68 ** 2
t74 = t71 * t68
t77 = t71 ** 2
t89 = t77 ** 2
t93 = lax_cond(t66, 0.135e1, t64)
t94 = math.sqrt(math.pi)
t97 = math.erf(0.1e1 / t93 / 0.2e1)
t99 = t93 ** 2
t102 = math.exp(-0.1e1 / t99 / 0.4e1)
t113 = lax_cond(0.135e1 <= t64, 0.1e1 / t68 / 0.36e2 - 0.1e1 / t71 / 0.960e3 + 0.1e1 / t74 / 0.26880e5 - 0.1e1 / t77 / 0.829440e6 + 0.1e1 / t77 / t68 / 0.28385280e8 - 0.1e1 / t77 / t71 / 0.1073479680e10 + 0.1e1 / t77 / t74 / 0.44590694400e11 - 0.1e1 / t89 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t93 * (t94 * t97 + 0.2e1 * t93 * (t102 - 0.3e1 / 0.2e1 - 0.2e1 * t99 * (t102 - 0.1e1))))
t118 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t113 * t52)
res = 0.2e1 * t118
