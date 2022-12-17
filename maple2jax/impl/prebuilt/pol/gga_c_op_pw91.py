t2 = r0 + r1
t3 = 0.1e1 / t2
t4 = (r0 - r1) * t3
t5 = abs(t4)
t10 = jnp.logical_and(r0 <= p_a_dens_threshold, r1 <= p_a_dens_threshold)
t11 = jnp.logical_or(0.1e1 - t5 <= p_a_zeta_threshold, t10)
t14 = p_a_zeta_threshold - 0.1e1
t17 = -t14
t18 = jnp.where(0.1e1 - t4 <= p_a_zeta_threshold, t17, t4)
t19 = jnp.where(0.1e1 + t4 <= p_a_zeta_threshold, t14, t18)
t20 = t19 ** 2
t29 = jnp.where(0.2e1 * r1 * t3 <= p_a_zeta_threshold, t17, t4)
t30 = jnp.where(0.2e1 * r0 * t3 <= p_a_zeta_threshold, t14, t29)
t31 = 0.1e1 + t30
t35 = 3 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t38 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t41 = 4 ** (0.1e1 / 0.3e1)
t42 = t36 / t38 * t41
t43 = 2 ** (0.1e1 / 0.3e1)
t44 = t31 <= p_a_zeta_threshold
t45 = 0.1e1 - t30
t46 = t45 <= p_a_zeta_threshold
t47 = jnp.where(t46, t17, t30)
t48 = jnp.where(t44, t14, t47)
t51 = ((0.1e1 + t48) * t2) ** (0.1e1 / 0.3e1)
t54 = 6 ** (0.1e1 / 0.3e1)
t55 = math.pi ** 2
t56 = t55 ** (0.1e1 / 0.3e1)
t57 = t56 ** 2
t58 = 0.1e1 / t57
t59 = t54 * t58
t60 = r0 ** 2
t61 = r0 ** (0.1e1 / 0.3e1)
t62 = t61 ** 2
t64 = 0.1e1 / t62 / t60
t68 = math.exp(-0.25e2 / 0.6e1 * t59 * s0 * t64)
t76 = t54 ** 2
t79 = t76 / t56 / t55
t80 = s0 ** 2
t81 = t60 ** 2
t87 = 0.69444444444444444444e-5 * t79 * t80 / t61 / t81 / r0
t90 = t76 / t56
t91 = math.sqrt(s0)
t94 = t91 / t61 / r0
t97 = math.asinh(0.64963333333333333333e0 * t90 * t94)
t109 = jnp.where(t31 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t51 / (0.1e1 + ((0.2743e0 - 0.1508e0 * t68) * t54 * t58 * s0 * t64 / 0.24e2 - t87) / (0.1e1 + 0.16370833333333333333e-1 * t90 * t94 * t97 + t87)) / 0.9e1)
t114 = jnp.where(t44, t17, -t30)
t115 = jnp.where(t46, t14, t114)
t118 = ((0.1e1 + t115) * t2) ** (0.1e1 / 0.3e1)
t121 = r1 ** 2
t122 = r1 ** (0.1e1 / 0.3e1)
t123 = t122 ** 2
t125 = 0.1e1 / t123 / t121
t129 = math.exp(-0.25e2 / 0.6e1 * t59 * s2 * t125)
t137 = s2 ** 2
t138 = t121 ** 2
t144 = 0.69444444444444444444e-5 * t79 * t137 / t122 / t138 / r1
t146 = math.sqrt(s2)
t149 = t146 / t122 / r1
t152 = math.asinh(0.64963333333333333333e0 * t90 * t149)
t164 = jnp.where(t45 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t118 / (0.1e1 + ((0.2743e0 - 0.1508e0 * t129) * t54 * t58 * s2 * t125 / 0.24e2 - t144) / (0.1e1 + 0.16370833333333333333e-1 * t90 * t149 * t152 + t144)) / 0.9e1)
t165 = t109 + t164
t167 = jnp.where(t165 == 0.0e0, DBL_EPSILON, t165)
t171 = t167 ** 2
t172 = t171 ** 2
res = jnp.where(t11, 0, -0.25000000000000000000e0 * (0.1e1 - t20) * t2 * (0.360663084e1 / t167 + 0.5764e0) / (0.315815266717518096e2 / t172 + 0.150327320916243744e2 / t171 / t167 + 0.1788764629788e1 / t171))
