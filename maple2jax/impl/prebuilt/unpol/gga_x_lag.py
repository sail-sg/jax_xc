t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = t21 ** 2
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t27 = math.sqrt(s0)
t28 = 2 ** (0.1e1 / 0.3e1)
t34 = (t22 / t24 * t27 * t28 / t20 / r0) ** 0.2626712e1
t38 = (0.1e1 + 0.13471619689594796103e-3 * t34) ** (-0.657946e0)
t42 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.22554757207579166202e-4 * t3 / t4 * t18 * t20 * t34 * t38)
res = 0.2e1 * t42
