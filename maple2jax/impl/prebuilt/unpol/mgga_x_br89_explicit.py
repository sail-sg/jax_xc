t3 = 0.1e1 <= p_a_zeta_threshold
t4 = p_a_zeta_threshold - 0.1e1
t6 = lax_cond(t3, -t4, 0)
t7 = lax_cond(t3, t4, t6)
t8 = 0.1e1 + t7
t10 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t12 = t8 ** (0.1e1 / 0.3e1)
t14 = lax_cond(t8 <= p_a_zeta_threshold, t10 * p_a_zeta_threshold, t12 * t8)
t15 = r0 ** (0.1e1 / 0.3e1)
t18 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t21 = 4 ** (0.1e1 / 0.3e1)
t22 = math.pi ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t24 = 2 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t27 = t15 ** 2
t29 = 0.1e1 / t27 / r0
t37 = r0 ** 2
t43 = l0 * t25 * t29 / 0.6e1 - 0.2e1 / 0.3e1 * params_a_gamma * tau0 * t25 * t29 + params_a_gamma * s0 * t25 / t27 / t37 / 0.12e2
t44 = abs(t43)
t47 = lax_cond(0.0e0 < t43, 0.50e-12, -0.50e-12)
t48 = lax_cond(t44 < 0.50e-12, t47, t43)
t51 = 0.2e1 / 0.3e1 * t23 / t48
t54 = lax_cond(-0.50e-12 < t51, -0.50e-12, t51)
t57 = math.atan(0.15255251812009530e1 * t54 + 0.4576575543602858e0)
t60 = t54 ** 2
t62 = t60 * t54
t64 = t60 ** 2
t66 = t64 * t54
t79 = lax_cond(0.50e-12 < t51, t51, 0.50e-12)
t81 = math.asinh(0.1e1 / (0.2085749716493756e1 * t79))
t84 = t79 ** 2
t86 = t84 * t79
t88 = t84 ** 2
t90 = t88 * t79
t102 = lax_cond(t51 <= 0.0e0, (-t57 + 0.4292036732051034e0) * (0.7566445420735584e0 - 0.26363977871370960e1 * t54 + 0.54745159964232880e1 * t60 - 0.12657308127108290e2 * t62 + 0.41250584725121360e1 * t64 - 0.30425133957163840e2 * t66) / (0.4771976183772063e0 - 0.17799813494556270e1 * t54 + 0.38433841862302150e1 * t60 - 0.95912050880518490e1 * t62 + 0.21730180285916720e1 * t64 - 0.30425133851603660e2 * t66), (t81 + 0.2e1) * (0.4435009886795587e-4 + 0.58128653604457910e0 * t79 + 0.66742764515940610e2 * t84 + 0.43426780897229770e3 * t86 + 0.8247765766052239000e3 * t88 + 0.16579652731582120e4 * t90) / (0.3347285060926091e-4 + 0.47917931023971350e0 * t79 + 0.62392268338574240e2 * t84 + 0.46314816427938120e3 * t86 + 0.7852360350104029000e3 * t88 + 0.1657962968223273000000e4 * t90))
t104 = math.exp(t102 / 0.3e1)
t106 = math.exp(-t102)
t116 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -t14 * t15 / t18 * t21 * t104 * (0.1e1 - t106 * (0.1e1 + t102 / 0.2e1)) / t102 / 0.4e1)
res = 0.2e1 * t116
