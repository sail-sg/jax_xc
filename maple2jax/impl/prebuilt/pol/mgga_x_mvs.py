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
t29 = r0 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t34 = r0 ** 2
t39 = tau0 / t30 / r0 - s0 / t30 / t34 / 0.8e1
t40 = 6 ** (0.1e1 / 0.3e1)
t42 = math.pi ** 2
t43 = t42 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t45 = 0.1e1 / t44
t50 = t39 ** 2
t52 = t40 ** 2
t54 = 0.1e1 / t43 / t42
t55 = t52 * t54
t59 = (0.1e1 + 0.25e2 / 0.81e2 * params_a_e1 * t50 * t55) ** 2
t60 = t50 ** 2
t62 = t42 ** 2
t65 = t40 / t44 / t62
t69 = (t59 + 0.1250e4 / 0.2187e4 * params_a_c1 * t60 * t65) ** (0.1e1 / 0.4e1)
t74 = params_a_b * t52
t75 = s0 ** 2
t77 = t34 ** 2
t85 = (0.1e1 + t74 * t54 * t75 / t29 / t77 / r0 / 0.576e3) ** (0.1e1 / 0.8e1)
t90 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + params_a_k0 * (0.1e1 - 0.5e1 / 0.9e1 * t39 * t40 * t45) / t69) / t85)
t92 = jnp.where(t10, t15, -t17)
t93 = jnp.where(t14, t11, t92)
t94 = 0.1e1 + t93
t96 = t94 ** (0.1e1 / 0.3e1)
t98 = jnp.where(t94 <= p_a_zeta_threshold, t23, t96 * t94)
t100 = r1 ** (0.1e1 / 0.3e1)
t101 = t100 ** 2
t105 = r1 ** 2
t110 = tau1 / t101 / r1 - s2 / t101 / t105 / 0.8e1
t116 = t110 ** 2
t121 = (0.1e1 + 0.25e2 / 0.81e2 * params_a_e1 * t116 * t55) ** 2
t122 = t116 ** 2
t127 = (t121 + 0.1250e4 / 0.2187e4 * params_a_c1 * t122 * t65) ** (0.1e1 / 0.4e1)
t132 = s2 ** 2
t134 = t105 ** 2
t142 = (0.1e1 + t74 * t54 * t132 / t100 / t134 / r1 / 0.576e3) ** (0.1e1 / 0.8e1)
t147 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t98 * t28 * (0.1e1 + params_a_k0 * (0.1e1 - 0.5e1 / 0.9e1 * t110 * t40 * t45) / t127) / t142)
res = t90 + t147
