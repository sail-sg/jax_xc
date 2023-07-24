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
t44 = math.exp(-t41 / 0.24e2)
t48 = t29 ** 2
t51 = t48 / t31 / t30
t52 = s0 ** 2
t53 = t35 ** 2
t61 = math.log(0.1e1 + 0.13780328706878157639e-4 * t51 * t52 / t36 / t53 / r0)
t69 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t41 + 0.40024242767108462450e-2 * t34 * t40 * t44 + t61)))
t71 = lax_cond(t10, t15, -t17)
t72 = lax_cond(t14, t11, t71)
t73 = 0.1e1 + t72
t75 = t73 ** (0.1e1 / 0.3e1)
t77 = lax_cond(t73 <= p_a_zeta_threshold, t23, t75 * t73)
t79 = r1 ** 2
t80 = r1 ** (0.1e1 / 0.3e1)
t81 = t80 ** 2
t84 = s2 / t81 / t79
t85 = t34 * t84
t88 = math.exp(-t85 / 0.24e2)
t92 = s2 ** 2
t93 = t79 ** 2
t101 = math.log(0.1e1 + 0.13780328706878157639e-4 * t51 * t92 / t80 / t93 / r1)
t109 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t77 * t27 * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t85 + 0.40024242767108462450e-2 * t34 * t84 * t88 + t101)))
res = t69 + t109
