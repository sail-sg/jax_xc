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
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t43 = t29 ** 2
t46 = t43 / t31 / t30
t47 = s0 ** 2
t48 = t35 ** 2
t62 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t34 * s0 / t37 / t35 + 0.32911784453572541027e-4 * t46 * t47 / t36 / t48 / r0)))
t64 = jnp.where(t10, t15, -t17)
t65 = jnp.where(t14, t11, t64)
t66 = 0.1e1 + t65
t68 = t66 ** (0.1e1 / 0.3e1)
t70 = jnp.where(t66 <= p_a_zeta_threshold, t23, t68 * t66)
t72 = r1 ** 2
t73 = r1 ** (0.1e1 / 0.3e1)
t74 = t73 ** 2
t80 = s2 ** 2
t81 = t72 ** 2
t95 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t70 * t27 * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t34 * s2 / t74 / t72 + 0.32911784453572541027e-4 * t46 * t80 / t73 / t81 / r1)))
res = t62 + t95
