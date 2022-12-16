t2 = r0 + r1
t3 = 0.1e1 / t2
t6 = 0.2e1 * r0 * t3 <= p_a_zeta_threshold
t7 = p_a_zeta_threshold - 0.1e1
t10 = 0.2e1 * r1 * t3 <= p_a_zeta_threshold
t11 = -t7
t13 = (r0 - r1) * t3
t14 = jnp.where(t10, t11, t13)
t15 = jnp.where(t6, t7, t14)
t16 = 0.1e1 + t15
t18 = math.sqrt(p_a_zeta_threshold)
t19 = t18 * p_a_zeta_threshold
t20 = math.sqrt(t16)
t22 = jnp.where(t16 <= p_a_zeta_threshold, t19, t20 * t16)
t24 = math.sqrt(0.2e1)
t25 = math.sqrt(t2)
t26 = t24 * t25
t27 = r0 ** 2
t28 = 0.1e1 / t27
t37 = 0.1e1 / math.pi
t38 = (l0 * t28 / 0.4e1 - tau0 * t28 + s0 / t27 / r0 / 0.8e1) * t37
t40 = jnp.where(-0.9999999999e0 < t38, t38, -0.9999999999e0)
t41 = math.exp(-1)
t43 = scipy.special.lambertw(t40 * t41)
t46 = scipy.special.i0(t43 / 0.2e1 + 0.1e1 / 0.2e1)
t50 = jnp.where(r0 <= p_a_dens_threshold, 0, -math.pi * t22 * t26 * t46 / 0.8e1)
t52 = jnp.where(t6, t11, -t13)
t53 = jnp.where(t10, t7, t52)
t54 = 0.1e1 + t53
t56 = math.sqrt(t54)
t58 = jnp.where(t54 <= p_a_zeta_threshold, t19, t56 * t54)
t60 = r1 ** 2
t61 = 0.1e1 / t60
t70 = (l1 * t61 / 0.4e1 - tau1 * t61 + s2 / t60 / r1 / 0.8e1) * t37
t72 = jnp.where(-0.9999999999e0 < t70, t70, -0.9999999999e0)
t74 = scipy.special.lambertw(t72 * t41)
t77 = scipy.special.i0(t74 / 0.2e1 + 0.1e1 / 0.2e1)
t81 = jnp.where(r1 <= p_a_dens_threshold, 0, -math.pi * t58 * t26 * t77 / 0.8e1)
res = t50 + t81
