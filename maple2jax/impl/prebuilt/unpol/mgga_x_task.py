t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = t11 + 0.1e1
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t32 = t31 * t30
t36 = t21 / t24 * s0 * t28 / t32 / 0.24e2
t37 = 0.0e0 < t36
t38 = lax_cond(t37, t36, 0)
t39 = t38 ** (0.1e1 / 0.4e1)
t42 = math.exp(-params_a_task_c / t39)
t44 = lax_cond(t37, 0.1e1 - t42, 0)
t46 = params_a_task_bnu[0]
t47 = t30 ** 2
t49 = t31 * t47 * t30
t52 = t47 * r0
t54 = r0 * tau0
t57 = 0.1e1 / r0
t59 = 0.1e1 / tau0
t68 = lax_cond(0.0e0 < 0.12500000000000000000e0 * (0.79999999992000000000e1 * t54 - s0) * t57 * t59, (0.8e1 * t54 - s0) * t57 * t59 / 0.8e1, 0.1e-9)
t69 = tau0 * t68
t73 = t19 * t30 * r0
t75 = tau0 ** 2
t76 = t68 ** 2
t77 = t75 * t76
t80 = t31 * r0
t84 = t75 * tau0 * t76 * t68
t87 = t75 ** 2
t89 = t76 ** 2
t92 = params_a_task_bnu[1]
t104 = params_a_task_bnu[2]
t113 = 0.47049607861172565388e8 * t46 * t49 + 0.65546183777551744717e8 * t46 * t52 * t69 + 0.34242864099506828874e8 * t46 * t73 * t77 + 0.79507891258050066525e7 * t46 * t80 * t84 + 0.69227979374755602348e6 * t46 * t87 * t89 - 0.47049607861172565388e8 * t92 * t49 - 0.32773091888775872359e8 * t92 * t52 * t69 + 0.39753945629025033263e7 * t92 * t80 * t84 + 0.69227979374755602348e6 * t92 * t87 * t89 + 0.47049607861172565386e8 * t104 * t49 - 0.65546183777551744717e8 * t104 * t52 * t69 - 0.57071440165844714789e8 * t104 * t73 * t77
t120 = params_a_task_bnu[3]
t135 = params_a_task_bnu[4]
t150 = -0.79507891258050066526e7 * t104 * t80 * t84 + 0.69227979374755602349e6 * t104 * t87 * t89 - 0.47049607861172565384e8 * t120 * t49 + 0.22941164322143110651e9 * t120 * t52 * t69 - 0.4e-11 * t120 * t73 * t77 - 0.27827761940317523285e8 * t120 * t80 * t84 + 0.69227979374755602348e6 * t120 * t87 * t89 + 0.4704960786117256539e8 * t135 * t49 - 0.45882328644286221302e9 * t135 * t52 * t69 + 0.39950008116091300353e9 * t135 * t73 * t77 - 0.55655523880635046568e8 * t135 * t80 * t84 + 0.6922797937475560235e6 * t135 * t87 * t89
t155 = (0.82820720060468819300e2 * t80 + 0.28844991406148167646e2 * t69) ** 2
t156 = t155 ** 2
t161 = t4 ** 2
t163 = t19 * t52 * t161 * t22
t164 = params_a_task_anu[0]
t167 = params_a_task_anu[1]
t170 = params_a_task_anu[2]
t173 = t3 * t32
t174 = t4 * math.pi
t175 = t174 * s0
t182 = t3 ** 2
t183 = s0 ** 2
t184 = t182 * t183
t193 = (t3 * s0 + 0.12e2 * t174 * t32) ** 2
t198 = t44 ** params_a_task_d
t204 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params_a_task_h0x * t44 + (0.10e1 - (t113 + t150) / t156) * ((0.24e2 * t173 * t175 * t164 - 0.72e2 * t173 * t175 * t170 + 0.144e3 * t163 * t164 - 0.144e3 * t163 * t167 + 0.144e3 * t163 * t170 + t184 * t164 + t184 * t167 + t184 * t170) / t193 - params_a_task_h0x) * t198))
res = 0.2e1 * t204
