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
t19 = jnp.where(t15, t16, t18)
t20 = jnp.where(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = jnp.where(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = 6 ** (0.1e1 / 0.3e1)
t34 = params_a_alpha * t33
t35 = math.pi ** 2
t36 = t35 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t38 = 0.1e1 / t37
t40 = r0 ** 2
t41 = r0 ** (0.1e1 / 0.3e1)
t42 = t41 ** 2
t45 = t38 * s0 / t42 / t40
t48 = math.exp(-t34 * t45 / 0.24e2)
t54 = t33 ** 2
t55 = 0.1e1 / t36
t56 = t54 * t55
t57 = math.sqrt(s0)
t59 = 0.1e1 / t41 / r0
t63 = (t56 * t57 * t59 / 0.12e2) ** params_a_expo
t64 = params_a_f * t63
t68 = params_a_b * t54
t73 = math.asinh(t68 * t55 * t57 * t59 / 0.12e2)
t84 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + ((params_a_d * t48 + params_a_c) * t33 * t45 / 0.24e2 - t64) / (0.1e1 + t56 * t57 * t59 * params_a_a * t73 / 0.12e2 + t64)))
t86 = jnp.where(t11, t16, -t18)
t87 = jnp.where(t15, t12, t86)
t88 = 0.1e1 + t87
t90 = t88 ** (0.1e1 / 0.3e1)
t91 = t90 ** 2
t93 = jnp.where(t88 <= p_a_zeta_threshold, t25, t91 * t88)
t96 = r1 ** 2
t97 = r1 ** (0.1e1 / 0.3e1)
t98 = t97 ** 2
t101 = t38 * s2 / t98 / t96
t104 = math.exp(-t34 * t101 / 0.24e2)
t110 = math.sqrt(s2)
t112 = 0.1e1 / t97 / r1
t116 = (t56 * t110 * t112 / 0.12e2) ** params_a_expo
t117 = params_a_f * t116
t125 = math.asinh(t68 * t55 * t110 * t112 / 0.12e2)
t136 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t93 * t31 * (0.1e1 + ((params_a_d * t104 + params_a_c) * t33 * t101 / 0.24e2 - t117) / (0.1e1 + t56 * t110 * t112 * params_a_a * t125 / 0.12e2 + t117)))
res = t84 + t136
