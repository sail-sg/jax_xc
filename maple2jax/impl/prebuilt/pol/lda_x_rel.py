t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t4 = 0.1e1 / t3
t5 = t2 * t4
t6 = r0 + r1
t7 = 0.1e1 / t6
t8 = r0 * t7
t11 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t12 = t11 * p_a_zeta_threshold
t13 = 2 ** (0.1e1 / 0.3e1)
t15 = t8 ** (0.1e1 / 0.3e1)
t19 = jnp.where(0.2e1 * t8 <= p_a_zeta_threshold, t12, 0.2e1 * t13 * r0 * t7 * t15)
t20 = t6 ** (0.1e1 / 0.3e1)
t24 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t19 * t20)
t26 = r1 * t7
t30 = t26 ** (0.1e1 / 0.3e1)
t34 = jnp.where(0.2e1 * t26 <= p_a_zeta_threshold, t12, 0.2e1 * t13 * r1 * t7 * t30)
t38 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t34 * t20)
t40 = 9 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t42 = t3 ** 2
t45 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t46 = t45 ** 2
t49 = t20 ** 2
t54 = math.sqrt(0.1e1 + 0.17750451365686221606e-4 * t41 * t42 * t2 / t46 * t49)
t63 = t2 ** 2
t69 = math.asinh(0.24324508467583486202e-2 * t40 * t3 * t63 / t45 * t20)
t79 = (0.15226222180972388889e2 * t54 * t41 * t4 * t2 * t45 / t20 - 0.20865405771390201384e4 * t69 * t40 / t42 * t63 * t46 / t49) ** 2
res = (t24 + t38) * (0.1e1 - 0.15e1 * t79)
