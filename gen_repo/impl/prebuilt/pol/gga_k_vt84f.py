t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t6 = t3 * t4 * math.pi
t7 = r0 + r1
t8 = 0.1e1 / t7
t11 = 0.2e1 * r0 * t8 <= p_a_zeta_threshold
t12 = p_a_zeta_threshold - 0.1e1
t15 = 0.2e1 * r1 * t8 <= p_a_zeta_threshold
t16 = -t12
t18 = (r0 - r1) * t8
t19 = lax_cond(t15, t16, t18)
t20 = lax_cond(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = lax_cond(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = 6 ** (0.1e1 / 0.3e1)
t34 = t33 ** 2
t35 = math.pi ** 2
t36 = t35 ** (0.1e1 / 0.3e1)
t38 = t34 / t36
t39 = math.sqrt(s0)
t40 = r0 ** (0.1e1 / 0.3e1)
t45 = t38 * t39 / t40 / r0 / 0.12e2
t46 = math.sqrt(DBL_EPSILON)
t49 = (-params_a_mu + params_a_alpha + 0.5e1 / 0.3e1) * t33
t50 = t36 ** 2
t51 = 0.1e1 / t50
t53 = r0 ** 2
t54 = t40 ** 2
t61 = params_a_mu ** 2
t63 = (params_a_mu * params_a_alpha + t61 - params_a_alpha) * t34
t65 = 0.1e1 / t36 / t35
t66 = s0 ** 2
t68 = t53 ** 2
t77 = lax_cond(t46 < t45, t45, t46)
t78 = t77 ** 2
t79 = params_a_mu * t78
t81 = math.exp(-params_a_alpha * t78)
t86 = t78 ** 2
t88 = math.exp(-params_a_alpha * t86)
t95 = lax_cond(t45 <= t46, 0.1e1 + t49 * t51 * s0 / t54 / t53 / 0.24e2 + t63 * t65 * t66 / t40 / t68 / r0 / 0.576e3, 0.1e1 - t79 * t81 / (0.1e1 + t79) + (0.1e1 - t88) * (0.1e1 / t78 - 0.1e1) + 0.5e1 / 0.3e1 * t78)
t99 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * t95)
t101 = lax_cond(t11, t16, -t18)
t102 = lax_cond(t15, t12, t101)
t103 = 0.1e1 + t102
t105 = t103 ** (0.1e1 / 0.3e1)
t106 = t105 ** 2
t108 = lax_cond(t103 <= p_a_zeta_threshold, t25, t106 * t103)
t110 = math.sqrt(s2)
t111 = r1 ** (0.1e1 / 0.3e1)
t116 = t38 * t110 / t111 / r1 / 0.12e2
t119 = r1 ** 2
t120 = t111 ** 2
t126 = s2 ** 2
t128 = t119 ** 2
t137 = lax_cond(t46 < t116, t116, t46)
t138 = t137 ** 2
t139 = params_a_mu * t138
t141 = math.exp(-params_a_alpha * t138)
t146 = t138 ** 2
t148 = math.exp(-params_a_alpha * t146)
t155 = lax_cond(t116 <= t46, 0.1e1 + t49 * t51 * s2 / t120 / t119 / 0.24e2 + t63 * t65 * t126 / t111 / t128 / r1 / 0.576e3, 0.1e1 - t139 * t141 / (0.1e1 + t139) + (0.1e1 - t148) * (0.1e1 / t138 - 0.1e1) + 0.5e1 / 0.3e1 * t138)
t159 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t108 * t31 * t155)
res = t99 + t159
