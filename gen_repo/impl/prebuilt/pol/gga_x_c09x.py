t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = lax_cond(t14, t15, t17)
t19 = lax_cond(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = lax_cond(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t40 = s0 / t37 / t35
t41 = t34 * t40
t43 = math.exp(-0.20125000000000000000e-2 * t41)
t48 = math.exp(-0.10062500000000000000e-2 * t41)
t54 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.2245e1 + 0.25708333333333333333e-2 * t34 * t40 * t43 - 0.1245e1 * t48))
t56 = lax_cond(t10, t15, -t17)
t57 = lax_cond(t14, t11, t56)
t58 = 0.1e1 + t57
t60 = t58 ** (0.1e1 / 0.3e1)
t62 = lax_cond(t58 <= p_a_zeta_threshold, t23, t60 * t58)
t64 = r1 ** 2
t65 = r1 ** (0.1e1 / 0.3e1)
t66 = t65 ** 2
t69 = s2 / t66 / t64
t70 = t34 * t69
t72 = math.exp(-0.20125000000000000000e-2 * t70)
t77 = math.exp(-0.10062500000000000000e-2 * t70)
t83 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t62 * t27 * (0.2245e1 + 0.25708333333333333333e-2 * t34 * t69 * t72 - 0.1245e1 * t77))
res = t54 + t83
