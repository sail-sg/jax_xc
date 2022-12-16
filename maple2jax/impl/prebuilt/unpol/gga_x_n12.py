t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = 0.1e1 + t11
t13 = t12 <= p_a_zeta_threshold
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t13, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t24 = 2 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t26 = r0 ** 2
t27 = t19 ** 2
t29 = 0.1e1 / t27 / t26
t34 = 0.1e1 + 0.4e-2 * s0 * t25 * t29
t36 = t25 * t29 / t34
t40 = s0 ** 2
t42 = t26 ** 2
t47 = t34 ** 2
t49 = t24 / t19 / t42 / r0 / t47
t53 = t40 * s0
t55 = t42 ** 2
t59 = 0.1e1 / t55 / t47 / t34
t80 = jnp.where(t13, 0.1e1 / t14, 0.1e1 / t16)
t83 = 0.1e1 + 0.39999999999999999998e0 / t19 * t24 * t80
t100 = t83 ** 2
t124 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params_a_CC_0_[0] + 0.4e-2 * params_a_CC_0_[1] * s0 * t36 + 0.32e-4 * params_a_CC_0_[2] * t40 * t49 + 0.256e-6 * params_a_CC_0_[3] * t53 * t59 + (params_a_CC_1_[0] + 0.4e-2 * params_a_CC_1_[1] * s0 * t36 + 0.32e-4 * params_a_CC_1_[2] * t40 * t49 + 0.256e-6 * params_a_CC_1_[3] * t53 * t59) / t83 + (params_a_CC_2_[0] + 0.4e-2 * params_a_CC_2_[1] * s0 * t36 + 0.32e-4 * params_a_CC_2_[2] * t40 * t49 + 0.256e-6 * params_a_CC_2_[3] * t53 * t59) / t100 + (params_a_CC_3_[0] + 0.4e-2 * params_a_CC_3_[1] * s0 * t36 + 0.32e-4 * params_a_CC_3_[2] * t40 * t49 + 0.256e-6 * params_a_CC_3_[3] * t53 * t59) / t100 / t83))
res = 0.2e1 * t124
