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
t30 = t29 ** 2
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t34 = t30 / t32
t35 = math.sqrt(s0)
t36 = r0 ** (0.1e1 / 0.3e1)
t39 = t35 / t36 / r0
t41 = t34 * t39 / 0.12e2
t42 = fd_int0(t41)
t43 = math.log(t41)
t45 = fd_int1(t41)
t54 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 - t34 * t39 * (t42 * t43 - t45) / 0.12e2))
t56 = jnp.where(t10, t15, -t17)
t57 = jnp.where(t14, t11, t56)
t58 = 0.1e1 + t57
t60 = t58 ** (0.1e1 / 0.3e1)
t62 = jnp.where(t58 <= p_a_zeta_threshold, t23, t60 * t58)
t64 = math.sqrt(s2)
t65 = r1 ** (0.1e1 / 0.3e1)
t68 = t64 / t65 / r1
t70 = t34 * t68 / 0.12e2
t71 = fd_int0(t70)
t72 = math.log(t70)
t74 = fd_int1(t70)
t83 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t62 * t27 * (0.1e1 - t34 * t68 * (t71 * t72 - t74) / 0.12e2))
res = t54 + t83
