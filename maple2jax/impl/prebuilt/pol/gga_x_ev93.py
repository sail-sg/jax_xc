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
t28 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = params_a_a1 * t29
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t41 = t34 * s0 / t38 / t36
t44 = t29 ** 2
t45 = params_a_a2 * t44
t47 = 0.1e1 / t32 / t31
t48 = s0 ** 2
t50 = t36 ** 2
t54 = t47 * t48 / t37 / t50 / r0
t57 = t31 ** 2
t58 = 0.1e1 / t57
t59 = params_a_a3 * t58
t61 = t50 ** 2
t63 = t48 * s0 / t61
t68 = params_a_b1 * t29
t71 = params_a_b2 * t44
t74 = params_a_b3 * t58
t82 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + t30 * t41 / 0.24e2 + t45 * t54 / 0.576e3 + t59 * t63 / 0.2304e4) / (0.1e1 + t68 * t41 / 0.24e2 + t71 * t54 / 0.576e3 + t74 * t63 / 0.2304e4))
t84 = jnp.where(t10, t15, -t17)
t85 = jnp.where(t14, t11, t84)
t86 = 0.1e1 + t85
t88 = t86 ** (0.1e1 / 0.3e1)
t90 = jnp.where(t86 <= p_a_zeta_threshold, t23, t88 * t86)
t93 = r1 ** 2
t94 = r1 ** (0.1e1 / 0.3e1)
t95 = t94 ** 2
t98 = t34 * s2 / t95 / t93
t101 = s2 ** 2
t103 = t93 ** 2
t107 = t47 * t101 / t94 / t103 / r1
t111 = t103 ** 2
t113 = t101 * s2 / t111
t129 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t90 * t28 * (0.1e1 + t30 * t98 / 0.24e2 + t45 * t107 / 0.576e3 + t59 * t113 / 0.2304e4) / (0.1e1 + t68 * t98 / 0.24e2 + t71 * t107 / 0.576e3 + t74 * t113 / 0.2304e4))
res = t82 + t129
