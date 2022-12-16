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
t18 = jnp.where(t14, t15, t17)
t19 = jnp.where(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = jnp.where(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = params_a_a[0]
t30 = params_a_a[1]
t31 = 6 ** (0.1e1 / 0.3e1)
t33 = math.pi ** 2
t34 = t33 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t37 = params_a_mu * t31 / t35
t38 = 0.1e1 / params_a_kappa
t40 = r0 ** 2
t41 = r0 ** (0.1e1 / 0.3e1)
t42 = t41 ** 2
t47 = t37 * t38 * s0 / t42 / t40 / 0.24e2
t50 = 0.1e1 - 0.1e1 / (0.1e1 + t47)
t52 = params_a_a[2]
t53 = t50 ** 2
t55 = params_a_a[3]
t58 = params_a_a[4]
t59 = t53 ** 2
t61 = params_a_a[5]
t64 = params_a_b[0]
t65 = params_a_b[1]
t66 = math.exp(-t47)
t67 = 0.1e1 - t66
t69 = params_a_b[2]
t70 = t67 ** 2
t72 = params_a_b[3]
t75 = params_a_b[4]
t76 = t70 ** 2
t78 = params_a_b[5]
t81 = t55 * t53 * t50 + t61 * t59 * t50 + t72 * t70 * t67 + t78 * t76 * t67 + t30 * t50 + t52 * t53 + t58 * t59 + t65 * t67 + t69 * t70 + t75 * t76 + t29 + t64
t85 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t81)
t87 = jnp.where(t10, t15, -t17)
t88 = jnp.where(t14, t11, t87)
t89 = 0.1e1 + t88
t91 = t89 ** (0.1e1 / 0.3e1)
t93 = jnp.where(t89 <= p_a_zeta_threshold, t23, t91 * t89)
t96 = r1 ** 2
t97 = r1 ** (0.1e1 / 0.3e1)
t98 = t97 ** 2
t103 = t37 * t38 * s2 / t98 / t96 / 0.24e2
t106 = 0.1e1 - 0.1e1 / (0.1e1 + t103)
t108 = t106 ** 2
t112 = t108 ** 2
t116 = math.exp(-t103)
t117 = 0.1e1 - t116
t119 = t117 ** 2
t123 = t119 ** 2
t127 = t55 * t108 * t106 + t61 * t112 * t106 + t72 * t119 * t117 + t78 * t123 * t117 + t30 * t106 + t52 * t108 + t58 * t112 + t65 * t117 + t69 * t119 + t75 * t123 + t29 + t64
t131 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t93 * t27 * t127)
res = t85 + t131
