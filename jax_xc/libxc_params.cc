#include "libxc/src/xc.h"
#include "visit_struct.hpp"
#include <array>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

typedef void (*register_fn)(py::module_ &);
static std::vector<register_fn> register_fns;
static bool Register(register_fn fn) {
  register_fns.push_back(fn);
  return true;
}

template <typename T> decltype(auto) ToNumpy(const T &a) {
  return py::array(std::array<int, 0>({}), &a);
}

template <typename T, size_t N> decltype(auto) ToNumpy(const T (&a)[N]) {
  return py::array(std::array<int, 1>({N}), a);
}

template <typename T, size_t N, size_t M>
decltype(auto) ToNumpy(const T (&a)[N][M]) {
  return py::array(std::array<int, 2>({N, M}), a);
}

#define REGISTER_PARAMS(STRUCT, ...)                                           \
  VISITABLE_STRUCT(STRUCT, __VA_ARGS__);                                       \
  auto STRUCT##_to_numpy(uint64_t func) {                                      \
    std::map<std::string, py::array> ret;                                      \
    visit_struct::for_each(                                                    \
        *reinterpret_cast<STRUCT *>(                                           \
            reinterpret_cast<xc_func_type *>(func)->params),                   \
        [&](const char *name, const auto &value) {                             \
          ret[name] = ToNumpy(value);                                          \
        });                                                                    \
    return ret;                                                                \
  }                                                                            \
  void STRUCT##_register(py::module_ &m) {                                     \
    m.def(#STRUCT "_to_numpy", &STRUCT##_to_numpy);                            \
  }                                                                            \
  static bool STRUCT##_registered = Register(STRUCT##_register);

/* gga_c_acgga */

/* gga_c_acggap */

/* gga_c_am05 */
typedef struct {
  double alpha, gamma;
} gga_c_am05_params;

REGISTER_PARAMS(gga_c_am05_params, alpha, gamma);

/* gga_c_bmk */
typedef struct {
  double c_ss[5], c_ab[5];
} gga_c_bmk_params;

REGISTER_PARAMS(gga_c_bmk_params, c_ss, c_ab);

/* gga_c_ccdf */
typedef struct {
  double c1;
  double c2;
  double c3;
  double c4;
  double c5;
} gga_c_ccdf_params;

/* gga_c_chachiyo */
typedef struct {
  double ap, bp, cp, af, bf, cf, h;
} gga_c_chachiyo_params;

/* gga_c_cs1 */

/* gga_c_ft97 */

/* gga_c_gapc */

/* gga_c_gaploc */

/* gga_c_hcth_a */

/* gga_c_lm */
typedef struct {
  double lm_f;
} gga_c_lm_params;

/* gga_c_lyp */
typedef struct {
  double a, b, c, d;
} gga_c_lyp_params;

/* gga_c_lypr */
typedef struct {
  double a;
  double b;
  double c;
  double d;
  double m1;
  double m2;
  double omega;
} gga_c_lypr_params;

/* gga_c_op_b88 */

/* gga_c_op_g96 */

/* gga_c_op_pbe */

/* gga_c_op_pw91 */

/* gga_c_op_xalpha */

/* gga_c_optc */
typedef struct {
  double c1, c2;
} gga_c_optc_params;

/* gga_c_p86 */
typedef struct {
  double malpha;
  double mbeta;
  double mgamma;
  double mdelta;
  double aa;
  double bb;
  double ftilde;
} gga_c_p86_params;

/* gga_c_p86vwn */
typedef struct {
  double malpha;
  double mbeta;
  double mgamma;
  double mdelta;
  double aa;
  double bb;
  double ftilde;
} gga_c_p86vwn_params;

/* gga_c_pbe */
typedef struct {
  double beta, gamma, BB;
} gga_c_pbe_params;

/* gga_c_pbe_erf_gws */
typedef struct {
  double beta, gamma, a_c, omega;
} gga_c_pbe_erf_gws_params;

/* gga_c_pbe_vwn */
typedef struct {
  double beta, gamma, BB;
} gga_c_pbe_vwn_params;

/* gga_c_pbeloc */

/* gga_c_pw91 */

/* gga_c_q2d */

/* gga_c_regtpss */

/* gga_c_revtca */

/* gga_c_scan_e0 */

/* gga_c_sg4 */

/* gga_c_sogga11 */
typedef struct {
  double sogga11_a[6], sogga11_b[6];
} gga_c_sogga11_params;

/* gga_c_tca */

/* gga_c_w94 */

/* gga_c_wi */
typedef struct {
  double a, b, c, d, k;
} gga_c_wi_params;

/* gga_c_wl */

/* gga_c_zpbeint */
typedef struct {
  double beta, alpha;
} gga_c_zpbeint_params;

/* gga_c_zvpbeint */
typedef struct {
  double beta, alpha, omega;
} gga_c_zvpbeint_params;

/* gga_c_zvpbeloc */

/* gga_k_apbe */
typedef struct {
  double kappa, mu;
  double lambda; /* parameter used in the Odashima & Capelle versions */
} gga_k_apbe_params;

/* gga_k_apbeint */
typedef struct {
  double kappa, alpha, muPBE, muGE;
} gga_k_apbeint_params;

/* gga_k_dk */
typedef struct {
  double aa[5], bb[5];
} gga_k_dk_params;

/* gga_k_exp4 */

/* gga_k_gds08 */

/* gga_k_lc94 */
typedef struct {
  double a, b, c, d, f, alpha, expo;
} gga_k_lc94_params;

/* gga_k_lgap */
typedef struct {
  double kappa;
  double mu[3];
} gga_k_lgap_params;

/* gga_k_lgap_ge */
typedef struct {
  double mu[3];
} gga_k_lgap_ge_params;

/* gga_k_lkt */
typedef struct {
  double a;
} gga_k_lkt_params;

/* gga_k_llp */
typedef struct {
  double beta, gamma;
} gga_k_llp_params;

/* gga_k_meyer */

/* gga_k_mpbe */
typedef struct {
  double a;
  double c1, c2, c3;
} gga_k_mpbe_params;

/* gga_k_ol1 */

/* gga_k_ol2 */
typedef struct {
  double aa, bb, cc;
} gga_k_ol2_params;

/* gga_k_pearson */

/* gga_k_pg */
typedef struct {
  double pg_mu;
} gga_k_pg_params;

/* gga_k_pw86 */

/* gga_k_rational_p */
typedef struct {
  double C2; /* prefactor for s^2 term */
  double p;  /* exponent */
} gga_k_rational_p_params;

/* gga_k_tflw */
typedef struct {
  double lambda, gamma;
} gga_k_tflw_params;

/* gga_k_thakkar */

/* gga_k_vt84f */
typedef struct {
  double mu;
  double alpha;
} gga_k_vt84f_params;

/* gga_x_2d_b86 */

/* gga_x_2d_b86_mgc */

/* gga_x_2d_b88 */

/* gga_x_2d_pbe */

/* gga_x_airy */

/* gga_x_ak13 */
typedef struct {
  double B1, B2;
} gga_x_ak13_params;

/* gga_x_am05 */
typedef struct {
  double alpha, c;
} gga_x_am05_params;

/* gga_x_b86 */
typedef struct {
  double beta, gamma, omega;
} gga_x_b86_params;

/* gga_x_b88 */
typedef struct {
  double beta, gamma;
} gga_x_b88_params;

/* gga_x_bayesian */

/* gga_x_beefvdw */

/* gga_x_bpccac */

/* gga_x_c09x */

/* gga_x_cap */
typedef struct {
  double alphaoAx, c;
} gga_x_cap_params;

/* gga_x_chachiyo */

/* gga_x_dk87 */
typedef struct {
  double a1, b1, alpha;
} gga_x_dk87_params;

/* gga_x_ev93 */
typedef struct {
  double a1, a2, a3; /* numerator */
  double b1, b2, b3; /* denominator */
} gga_x_ev93_params;

/* gga_x_fd_lb94 */
typedef struct {
  double beta; /* screening parameter beta */
} gga_x_fd_lb94_params;

/* gga_x_ft97 */
typedef struct {
  double beta0, beta1, beta2;
} gga_x_ft97_params;

/* gga_x_g96 */

/* gga_x_gg99 */

/* gga_x_hcth_a */

/* gga_x_herman */

/* gga_x_hjs */
typedef struct {
  double a[6], b[9]; /* pointers to the a and b parameters */
} gga_x_hjs_params;

/* gga_x_hjs_b88_v2 */
typedef struct {
  double a[6], b[9]; /* pointers to the a and b parameters */
} gga_x_hjs_b88_v2_params;

/* gga_x_htbs */

/* gga_x_ityh */

/* gga_x_ityh_optx */
typedef struct {
  double a, b, gamma;
} gga_x_ityh_optx_params;

/* gga_x_ityh_pbe */
typedef struct {
  double kappa, mu;
  double lambda; /* parameter used in the Odashima & Capelle versions */
} gga_x_ityh_pbe_params;

/* gga_x_kt */
typedef struct {
  double gamma, delta;
} gga_x_kt_params;

/* gga_x_lag */

/* gga_x_lb */
typedef struct {
  double alpha;
  double beta;
  double gamma;
} gga_x_lb_params;

/* gga_x_lcgau */

/* gga_x_lg93 */

/* gga_x_lspbe */
typedef struct {
  double kappa; /* PBE kappa parameter */
  double mu;    /* PBE mu parameter */
  double alpha; /* alpha parameter, solved automatically */
} gga_x_lspbe_params;

/* gga_x_lsrpbe */
typedef struct {
  double kappa;
  double mu;
  double alpha;
} gga_x_lsrpbe_params;

/* gga_x_lv_rpw86 */

/* gga_x_mpbe */
typedef struct {
  double a;
  double c1, c2, c3;
} gga_x_mpbe_params;

/* gga_x_n12 */
typedef struct {
  double CC[4][4];
} gga_x_n12_params;

/* gga_x_ncap */
typedef struct {
  double alpha, beta, mu, zeta;
} gga_x_ncap_params;

/* gga_x_ol2 */
typedef struct {
  double aa, bb, cc;
} gga_x_ol2_params;

/* gga_x_optx */
typedef struct {
  double a, b, gamma;
} gga_x_optx_params;

/* gga_x_pbe */
typedef struct {
  double kappa, mu;
  double lambda; /* parameter used in the Odashima & Capelle versions */
} gga_x_pbe_params;

/* gga_x_pbe_erf_gws */
typedef struct {
  double kappa, b_PBE, ax, omega;
} gga_x_pbe_erf_gws_params;

/* gga_x_pbea */

/* gga_x_pbeint */
typedef struct {
  double kappa, alpha, muPBE, muGE;
} gga_x_pbeint_params;

/* gga_x_pbepow */

/* gga_x_pbetrans */

/* gga_x_pw86 */
typedef struct {
  double aa, bb, cc;
} gga_x_pw86_params;

/* gga_x_pw91 */
typedef struct {
  double a, b, c, d, f, alpha, expo;
} gga_x_pw91_params;

/* gga_x_q1d */

/* gga_x_q2d */

/* gga_x_rge2 */

/* gga_x_rpbe */
typedef struct {
  double rpbe_kappa, rpbe_mu;
} gga_x_rpbe_params;

/* gga_x_s12 */
typedef struct {
  double A, B, C, D, E;
  double bx;
} gga_x_s12_params;

/* gga_x_sfat */

/* gga_x_sfat_pbe */

/* gga_x_sg4 */

/* gga_x_sogga11 */
typedef struct {
  double kappa, mu, a[6], b[6];
} gga_x_sogga11_params;

/* gga_x_ssb_sw */
typedef struct {
  double A, B, C, D, E;
} gga_x_ssb_sw_params;

/* gga_x_vmt */
typedef struct {
  double mu;
  double alpha;
} gga_x_vmt_params;

/* gga_x_vmt84 */
typedef struct {
  double mu;
  double alpha;
} gga_x_vmt84_params;

/* gga_x_wc */

/* gga_x_wpbeh */

/* gga_xc_1w */

/* gga_xc_b97 */
typedef struct {
  double c_x[5], c_ss[5], c_ab[5];
} gga_xc_b97_params;

/* gga_xc_edf1 */

/* gga_xc_oblyp_d */

/* gga_xc_th1 */
typedef struct {
  double omega[21];
} gga_xc_th1_params;

/* gga_xc_th2 */

/* gga_xc_th3 */
typedef struct {
  double omega[19];
} gga_xc_th3_params;

/* gga_xc_vv10 */

/* hyb_gga_x_cam_s12 */
typedef struct {
  double A, B, C, D, E;
} hyb_gga_x_cam_s12_params;

/* hyb_gga_xc_b1wc */

/* hyb_gga_xc_b2plyp */

/* hyb_gga_xc_b3lyp */

/* hyb_gga_xc_cam_b3lyp */

/* hyb_gga_xc_cam_o3lyp */

/* hyb_gga_xc_camy_b3lyp */

/* hyb_gga_xc_camy_blyp */

/* hyb_gga_xc_case21 */
typedef struct {
  /* set statically */
  int k;            /* order of B splines */
  int Nsp;          /* number of B splines */
  double knots[14]; /* knot sequence */

  /* adjustable parameters */
  double cx[10]; /* exchange enhancement */
  double cc[10]; /* correlation enhancement */
  double gammax; /* gamma, exchange */
  double gammac; /* gamma, correlation */
  double ax;     /* fraction of exact exchange */
} hyb_gga_xc_case21_params;

/* hyb_gga_xc_edf2 */

/* hyb_gga_xc_hse */

/* hyb_gga_xc_lc_blyp */

/* hyb_gga_xc_lcy_blyp */

/* hyb_gga_xc_lcy_pbe */

/* hyb_gga_xc_o3lyp */

/* hyb_gga_xc_pbe_dh */

/* hyb_gga_xc_pbeh */

/* hyb_gga_xc_src1_blyp */

/* hyb_gga_xc_wb97 */
typedef struct {
  double c_x[5], c_ss[5], c_ab[5];
} gga_xc_wb97_params;

/* hyb_lda_xc_bn05 */

/* hyb_lda_xc_cam_lda0 */

/* hyb_mgga_x_dldf */

/* hyb_mgga_x_js18 */

/* hyb_mgga_x_m05 */
typedef struct {
  const double a[12];
  double csi_HF;
  double cx;
} mgga_x_m05_params;

/* hyb_mgga_x_mvsh */

/* hyb_mgga_x_pjs18 */

/* hyb_mgga_xc_b88b95 */

/* hyb_mgga_xc_br3p86 */

/* hyb_mgga_xc_kcis */

/* hyb_mgga_xc_tpssh */

/* hyb_mgga_xc_wb97mv */
typedef struct {
  double c_x[3], c_ss[5], c_os[6];
} hyb_mgga_xc_wb97_mv_params;

/* hybrids */

/* integrate */

/* lda */

/* lda_c_1d_csc */
typedef struct {
  double para[10], ferro[10];

  int interaction; /* 0: exponentially screened; 1: soft-Coulomb */
  double bb;       /* screening parameter */

} lda_c_1d_csc_params;

/* lda_c_1d_loos */

/* lda_c_2d_amgb */

/* lda_c_2d_prm */
typedef struct {
  double N;
  double c;
} lda_c_2d_prm_params;

/* lda_c_chachiyo */
typedef struct {
  double ap, bp, cp, af, bf, cf;
} lda_c_chachiyo_params;

/* lda_c_chachiyo_mod */
typedef struct {
  double ap, bp, cp, af, bf, cf;
} lda_c_chachiyo_mod_params;

/* lda_c_gk72 */

/* lda_c_gombas */

/* lda_c_hl */
typedef struct {
  double hl_r[2], hl_c[2];
} lda_c_hl_params;

/* lda_c_lp96 */
typedef struct {
  double C1, C2, C3;
} lda_c_lp96_params;

/* lda_c_ml1 */
typedef struct {
  double fc, q;
} lda_c_ml1_params;

/* lda_c_pk09 */

/* lda_c_pmgb06 */

/* lda_c_pw */
typedef struct {
  double pp[3], a[3], alpha1[3];
  double beta1[3], beta2[3], beta3[3], beta4[3];
  double fz20;
} lda_c_pw_params;

/* lda_c_pw_erf */

/* lda_c_pz */
typedef struct {
  double gamma[2];
  double beta1[2];
  double beta2[2];
  double a[2], b[2], c[2], d[2];
} lda_c_pz_params;

/* lda_c_rc04 */

/* lda_c_rpa */

/* lda_c_vwn */

/* lda_c_vwn_1 */

/* lda_c_vwn_2 */

/* lda_c_vwn_3 */

/* lda_c_vwn_4 */

/* lda_c_vwn_rpa */

/* lda_c_w20 */

/* lda_c_wigner */
typedef struct {
  double a, b;
} lda_c_wigner_params;

/* lda_k_gds08_worker */
typedef struct {
  double A, B, C;
} lda_k_gds08_params;

/* lda_k_tf */
typedef struct {
  double ax;
} lda_k_tf_params;

/* lda_k_zlp */

/* lda_x */
typedef struct {
  double alpha; /* parameter for Xalpha functional */
} lda_x_params;

/* lda_x_1d_exponential */
typedef struct {
  double beta; /* screening parameter beta */
} lda_x_1d_exponential_params;

/* lda_x_1d_soft */
typedef struct {
  double beta; /* screening parameter beta */
} lda_x_1d_soft_params;

/* lda_x_2d */

/* lda_x_erf */

/* lda_x_rel */

/* lda_x_sloc */
typedef struct {
  double a; /* prefactor */
  double b; /* exponent */
} lda_x_sloc_params;

/* lda_x_yukawa */

/* lda_xc_1d_ehwlrg */
typedef struct {
  double alpha;
  double a1, a2, a3;
} lda_xc_1d_ehwlrg_params;

/* lda_xc_ksdt */
typedef struct {
  double T;          /* In units of k_B */
  double thetaParam; /* This takes into account the difference between t and
                        theta_0 */

  double b[2][5], c[2][3], d[2][5], e[2][5];
} lda_xc_ksdt_params;

/* lda_xc_teter93 */

/* lda_xc_tih */

/* lda_xc_zlp */

/* math_brent */

/* mgga */

/* mgga_c_b88 */

/* mgga_c_b94 */
typedef struct {
  double gamma; /* gamma parameter in mgga_x_br89 */
  double css;   /* same-spin constant */
  double cab;   /* opposite-spin constant */
} mgga_c_b94_params;

/* mgga_c_bc95 */
typedef struct {
  double css, copp;
} mgga_c_bc95_params;

/* mgga_c_cc */

/* mgga_c_ccalda */
typedef struct {
  double c; /* parameter in eq 10 */
} mgga_c_ccalda_params;

/* mgga_c_cs */

/* mgga_c_kcis */

/* mgga_c_kcisk */

/* mgga_c_ltapw */
typedef struct {
  double ltafrac;
} mgga_c_ltapw_params;

/* mgga_c_m05 */
typedef struct {
  double gamma_ss, gamma_ab;
  const double css[5], cab[5];
  double Fermi_D_cnst; /* correction term similar to 10.1063/1.2800011 */
} mgga_c_m05_params;

/* mgga_c_m06l */
typedef struct {
  double gamma_ss, gamma_ab, alpha_ss, alpha_ab;
  const double css[5], cab[5], dss[6], dab[6];
  double Fermi_D_cnst; /* correction term similar to 10.1063/1.2800011 */
} mgga_c_m06l_params;

/* mgga_c_m08 */
typedef struct {
  const double m08_a[12], m08_b[12];
} mgga_c_m08_params;

/* mgga_c_pkzb */

/* mgga_c_r2scan */
typedef struct {
  double eta; /* regularization parameter */
} mgga_c_r2scan_params;

/* mgga_c_r2scanl */

/* mgga_c_revscan */

/* mgga_c_revtpss */
typedef struct {
  double d;
  double C0_c[4];
} mgga_c_revtpss_params;

/* mgga_c_rmggac */

/* mgga_c_rppscan */
typedef struct {
  double eta; /* regularization parameter */
} mgga_c_rppscan_params;

/* mgga_c_rregtm */

/* mgga_c_rscan */

/* mgga_c_scan */

/* mgga_c_scanl */

/* mgga_c_tpss */
typedef struct {
  double beta, d;
  double C0_c[4];
} mgga_c_tpss_params;

/* mgga_c_tpssloc */

/* mgga_c_vsxc */
typedef struct {
  const double alpha_ss, alpha_ab;
  const double dss[6], dab[6];
} mgga_c_vsxc_params;

/* mgga_k_csk */
typedef struct {
  double csk_a;
} mgga_k_csk_params;

/* mgga_k_csk_loc */
typedef struct {
  double csk_a, csk_cp, csk_cq;
} mgga_k_csk_loc_params;

/* mgga_k_gea2 */

/* mgga_k_gea4 */

/* mgga_k_lk */
typedef struct {
  double kappa;
} mgga_k_lk_params;

/* mgga_k_pc07 */
typedef struct {
  double a, b;
} mgga_k_pc07_params;

/* mgga_k_pgslb */
typedef struct {
  double pgslb_mu, pgslb_beta;
} mgga_k_pgslb_params;

/* mgga_k_rda */
typedef struct {
  double A0, A1, A2, A3;
  double beta1, beta2, beta3;
  double a, b, c;
} mgga_k_rda_params;

/* mgga_x_2d_js17 */

/* mgga_x_2d_prhg07 */

/* mgga_x_2d_prp10 */

/* mgga_x_br89 */
typedef struct {
  double gamma, at;
} mgga_x_br89_params;

/* mgga_x_br89_explicit */
typedef struct {
  double gamma;
} mgga_x_br89_explicit_params;

/* mgga_x_edmgga */

/* mgga_x_ft98 */
typedef struct {
  double a;
  double b;
  double a1;
  double a2;
  double b1;
  double b2;
} mgga_x_ft98_params;

/* mgga_x_gdme */
typedef struct {
  double a, AA, BB;
} mgga_x_gdme_params;

/* mgga_x_gvt4 */

/* mgga_x_gx */
typedef struct {
  double c0, c1, alphainf;
} mgga_x_gx_params;

/* mgga_x_jk */
typedef struct {
  double beta;
  double gamma;
} mgga_x_jk_params;

/* mgga_x_lta */
typedef struct {
  double ltafrac;
} mgga_x_lta_params;

/* mgga_x_m06l */
typedef struct {
  double a[12], d[6];
} mgga_x_m06l_params;

/* mgga_x_m08 */
typedef struct {
  const double a[12], b[12];
} mgga_x_m08_params;

/* mgga_x_m11 */
typedef struct {
  const double a[12], b[12];
} mgga_x_m11_params;

/* mgga_x_m11_l */
typedef struct {
  const double a[12], b[12], c[12], d[12];
} mgga_x_m11_l_params;

/* mgga_x_mbeef */

/* mgga_x_mbeefvdw */

/* mgga_x_mbr */
typedef struct {
  double gamma, beta, lambda;
} mgga_x_mbr_params;

/* mgga_x_mbrxc_bg */

/* mgga_x_mbrxh_bg */

/* mgga_x_mcml */

/* mgga_x_mggac */

/* mgga_x_mn12 */
typedef struct {
  const double c[40];
} mgga_x_mn12_params;

/* mgga_x_ms */
typedef struct {
  double kappa, c, b;
} mgga_x_ms_params;

/* mgga_x_msb */
typedef struct {
  double kappa, c, b;
} mgga_x_msb_params;

/* mgga_x_mvs */
typedef struct {
  double e1, c1, k0, b;
} mgga_x_mvs_params;

/* mgga_x_mvsb */
typedef struct {
  double e1, c1, k0, b;
} mgga_x_mvsb_params;

/* mgga_x_pbe_gx */

/* mgga_x_pkzb */

/* mgga_x_r2scan */
typedef struct {
  double c1, c2, d, k1;
  double eta, dp2;
} mgga_x_r2scan_params;

/* mgga_x_r2scanl */

/* mgga_x_r4scan */
typedef struct {
  double c1, c2, d, k1, eta;
  double dp2, dp4, da4;
} mgga_x_r4scan_params;

/* mgga_x_regtm */

/* mgga_x_regtpss */

/* mgga_x_revtm */

/* mgga_x_rlda */
typedef struct {
  double prefactor;
} mgga_x_rlda_params;

/* mgga_x_rppscan */
typedef struct {
  double c2, d, k1, eta;
} mgga_x_rppscan_params;

/* mgga_x_rscan */
typedef struct {
  double c2, d, k1;
  double taur, alphar;
} mgga_x_rscan_params;

/* mgga_x_rtpss */
typedef struct {
  double b, c, e, kappa, mu;
} mgga_x_rtpss_params;

/* mgga_x_sa_tpss */

/* mgga_x_scan */
typedef struct {
  double c1, c2, d, k1;
} mgga_x_scan_params;
typedef struct {
  double exx;
} hyb_mgga_x_scan0_params;

/* mgga_x_scanl */

/* mgga_x_task */
typedef struct {
  double task_c, task_d, task_h0x;
  double task_anu[3], task_bnu[5];
} mgga_x_task_params;

/* mgga_x_tau_hcth */
typedef struct {
  double cx_local[4];
  double cx_nlocal[4];
} mgga_x_tau_hcth_params;

/* mgga_x_tb09 */
typedef struct {
  double c;
  double alpha;
} mgga_x_tb09_params;

/* mgga_x_th */

/* mgga_x_tm */

/* mgga_x_tpss */
typedef struct {
  double b, c, e, kappa, mu;
  double BLOC_a, BLOC_b;
} mgga_x_tpss_params;

/* mgga_x_vcml */

/* mgga_x_vt84 */

/* mgga_xc_b97mv */
typedef struct {
  double c_x[5], c_ss[5], c_os[5];
} mgga_xc_b97_mv_params;

/* mgga_xc_b98 */

/* mgga_xc_cc06 */

/* mgga_xc_hle17 */

/* mgga_xc_lp90 */

/* mgga_xc_otpss_d */

/* mgga_xc_zlp */

PYBIND11_MODULE(libxc_params, m) {
  m.doc() = "Utility to extract libxc params."; // optional module docstring
  for (register_fn f : register_fns) {
    f(m);
  }
}
