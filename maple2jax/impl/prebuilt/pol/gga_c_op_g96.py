t2 = r0 + r1
t3 = 0.1e1 / t2
t4 = (r0 - r1) * t3
t5 = abs(t4)
t10 = jnp.logical_and(r0 <= p_a_dens_threshold, r1 <= p_a_dens_threshold)
t11 = jnp.logical_or(0.1e1 - t5 <= p_a_zeta_threshold, t10)
t14 = p_a_zeta_threshold - 0.1e1
t17 = -t14
t18 = lax_cond(0.1e1 - t4 <= p_a_zeta_threshold, t17, t4)
t19 = lax_cond(0.1e1 + t4 <= p_a_zeta_threshold, t14, t18)
t20 = t19 ** 2
t29 = lax_cond(0.2e1 * r1 * t3 <= p_a_zeta_threshold, t17, t4)
t30 = lax_cond(0.2e1 * r0 * t3 <= p_a_zeta_threshold, t14, t29)
t31 = 0.1e1 + t30
t35 = 3 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t38 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t40 = t36 / t38
t41 = 4 ** (0.1e1 / 0.3e1)
t42 = t40 * t41
t43 = 2 ** (0.1e1 / 0.3e1)
t44 = t31 <= p_a_zeta_threshold
t45 = 0.1e1 - t30
t46 = t45 <= p_a_zeta_threshold
t47 = lax_cond(t46, t17, t30)
t48 = lax_cond(t44, t14, t47)
t51 = ((0.1e1 + t48) * t2) ** (0.1e1 / 0.3e1)
t54 = math.sqrt(s0)
t55 = r0 ** (0.1e1 / 0.3e1)
t58 = t54 / t55 / r0
t59 = math.sqrt(t58)
t69 = lax_cond(t31 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t51 / (0.1e1 + 0.2e1 / 0.1233e4 * t40 * t41 * t59 * t58) / 0.9e1)
t74 = lax_cond(t44, t17, -t30)
t75 = lax_cond(t46, t14, t74)
t78 = ((0.1e1 + t75) * t2) ** (0.1e1 / 0.3e1)
t81 = math.sqrt(s2)
t82 = r1 ** (0.1e1 / 0.3e1)
t85 = t81 / t82 / r1
t86 = math.sqrt(t85)
t96 = lax_cond(t45 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t78 / (0.1e1 + 0.2e1 / 0.1233e4 * t40 * t41 * t86 * t85) / 0.9e1)
t97 = t69 + t96
t99 = lax_cond(t97 == 0.0e0, DBL_EPSILON, t97)
t103 = t99 ** 2
t104 = t103 ** 2
res = lax_cond(t11, 0, -0.25000000000000000000e0 * (0.1e1 - t20) * t2 * (0.359628532e1 / t99 + 0.5764e0) / (0.312207199195441936e2 / t104 + 0.149037398922132448e2 / t103 / t99 + 0.1778517305052e1 / t103))
