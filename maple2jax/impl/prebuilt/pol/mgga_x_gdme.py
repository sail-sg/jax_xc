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
t31 = 2 ** (0.1e1 / 0.3e1)
t34 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t35 = 0.1e1 / t34
t36 = 4 ** (0.1e1 / 0.3e1)
t38 = math.pi ** 2
t39 = t38 ** (0.1e1 / 0.3e1)
t40 = t39 ** 2
t44 = 0.2e1 / 0.9e1 * (params_a_AA + 0.3e1 / 0.5e1 * params_a_BB) * t31 * t35 * t36 / t40
t46 = params_a_BB * t2 * t35
t47 = t31 ** 2
t48 = t36 * t47
t50 = 0.1e1 / t39 / t38
t51 = params_a_a ** 2
t52 = t51 - params_a_a + 0.1e1 / 0.2e1
t54 = r0 ** (0.1e1 / 0.3e1)
t55 = t54 ** 2
t57 = 0.1e1 / t55 / r0
t70 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t44 + t46 * t48 * t50 * (t52 * l0 * t57 - 0.2e1 * tau0 * t57) / 0.27e2))
t72 = lax_cond(t10, t15, -t17)
t73 = lax_cond(t14, t11, t72)
t74 = 0.1e1 + t73
t76 = t74 ** (0.1e1 / 0.3e1)
t78 = lax_cond(t74 <= p_a_zeta_threshold, t23, t76 * t74)
t81 = r1 ** (0.1e1 / 0.3e1)
t82 = t81 ** 2
t84 = 0.1e1 / t82 / r1
t97 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t78 * t27 * (t44 + t46 * t48 * t50 * (t52 * l1 * t84 - 0.2e1 * tau1 * t84) / 0.27e2))
res = t70 + t97
