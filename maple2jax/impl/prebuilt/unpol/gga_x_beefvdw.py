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
t26 = t21 / t24
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t33 = 0.1e1 / t31 / t30
t42 = t26 * s0 * t29 * t33 / (0.4e1 + t26 * s0 * t29 * t33 / 0.24e2)
t44 = t42 / 0.12e2 - 0.1e1
t45 = t44 ** 2
t46 = t45 ** 2
t47 = t46 * t44
t48 = t46 ** 2
t49 = t48 * t47
t50 = t48 ** 2
t53 = t48 * t46
t56 = t48 * t45
t59 = t45 * t44
t60 = t48 * t59
t63 = t48 * t44
t66 = t46 * t59
t71 = t46 * t45
t87 = 0.41355861880146538750e4 * t50 * t49 - 0.54277774626371860320e4 * t50 * t53 + 0.40074935854432390114e5 * t50 * t56 - 0.29150193011493262292e5 * t50 * t60 + 0.90365611108522808258e5 * t50 * t63 - 0.16114215399846280595e6 * t50 * t66 - 0.13204466182182150467e6 * t50 * t48 + 0.25589479526235334610e6 * t50 * t71 - 0.32352403136049329184e6 * t50 * t46 + 0.18078200670879145336e6 * t50 * t47 - 0.12981481812794983922e6 * t50 * t59 + 0.56174007979372666951e5 * t50 * t44 + 0.27967048856303053872e6 * t50 * t45 - 0.16837084139014120539e6 * t50 + 0.70504541869034010051e5 * t48 * t71
t103 = 0.11313514630621233134e1 - 0.10276426607863824397e5 * t48 * t66 - 0.2810240180568462990e4 * t49 + 0.22748997850816485208e4 * t60 - 0.20148245175625047025e5 * t53 + 0.37835396407252402359e4 * t56 - 0.44233229018433803622e3 * t48 - 0.61754786104528599731e3 * t63 + 0.86005730499279641299e2 * t66 - 0.72975787893717136018e1 * t47 + 0.30542034959315850168e2 * t71 - 0.69459735177638985466e0 * t46 - 0.38916037779196815969e0 * t45 + 0.52755620115589800943e0 * t59 + 0.37534251004296526981e-1 * t42
t108 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (t87 + t103))
res = 0.2e1 * t108
