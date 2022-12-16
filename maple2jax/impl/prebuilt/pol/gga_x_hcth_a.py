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
t29 = t2 ** 2
t31 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t34 = 4 ** (0.1e1 / 0.3e1)
t35 = t29 / t31 * t34
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t42 = math.sqrt(s0)
t45 = t42 / t37 / r0
t46 = math.asinh(t45)
t49 = 0.1e1 + 0.252e-1 * t45 * t46
t52 = t49 ** 2
t63 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.109878e1 + 0.93333333333333333332e-3 * t35 * s0 / t38 / t36 * (-0.251173e1 / t49 + 0.37198333333333333333e1 / t52)))
t65 = jnp.where(t10, t15, -t17)
t66 = jnp.where(t14, t11, t65)
t67 = 0.1e1 + t66
t69 = t67 ** (0.1e1 / 0.3e1)
t71 = jnp.where(t67 <= p_a_zeta_threshold, t23, t69 * t67)
t73 = r1 ** 2
t74 = r1 ** (0.1e1 / 0.3e1)
t75 = t74 ** 2
t79 = math.sqrt(s2)
t82 = t79 / t74 / r1
t83 = math.asinh(t82)
t86 = 0.1e1 + 0.252e-1 * t82 * t83
t89 = t86 ** 2
t100 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t71 * t27 * (0.109878e1 + 0.93333333333333333332e-3 * t35 * s2 / t75 / t73 * (-0.251173e1 / t86 + 0.37198333333333333333e1 / t89)))
res = t63 + t100
