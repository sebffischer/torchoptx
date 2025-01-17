// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "torchoptx_types.h"
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_torchoptx_sgd
torchoptx::optim_sgd rcpp_torchoptx_sgd(torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov);
RcppExport SEXP _torchoptx_rcpp_torchoptx_sgd(SEXP paramsSEXP, SEXP lrSEXP, SEXP momentumSEXP, SEXP dampeningSEXP, SEXP weight_decaySEXP, SEXP nesterovSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type momentum(momentumSEXP);
    Rcpp::traits::input_parameter< double >::type dampening(dampeningSEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< bool >::type nesterov(nesterovSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_torchoptx_sgd(params, lr, momentum, dampening, weight_decay, nesterov));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_torchoptx_sgd_step
void rcpp_torchoptx_sgd_step(torchoptx::optim_sgd opt);
RcppExport SEXP _torchoptx_rcpp_torchoptx_sgd_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_sgd >::type opt(optSEXP);
    rcpp_torchoptx_sgd_step(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_torchoptx_sgd_zero_grad
void rcpp_torchoptx_sgd_zero_grad(torchoptx::optim_sgd opt);
RcppExport SEXP _torchoptx_rcpp_torchoptx_sgd_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_sgd >::type opt(optSEXP);
    rcpp_torchoptx_sgd_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_torchoptx_adam
torchoptx::optim_adam rcpp_torchoptx_adam(torch::TensorList params, double lr, double betas0, double betas1, double eps, double weight_decay, bool amsgrad);
RcppExport SEXP _torchoptx_rcpp_torchoptx_adam(SEXP paramsSEXP, SEXP lrSEXP, SEXP betas0SEXP, SEXP betas1SEXP, SEXP epsSEXP, SEXP weight_decaySEXP, SEXP amsgradSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type betas0(betas0SEXP);
    Rcpp::traits::input_parameter< double >::type betas1(betas1SEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< bool >::type amsgrad(amsgradSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_torchoptx_adam(params, lr, betas0, betas1, eps, weight_decay, amsgrad));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_torchoptx_adam_step
void rcpp_torchoptx_adam_step(torchoptx::optim_adam opt);
RcppExport SEXP _torchoptx_rcpp_torchoptx_adam_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_adam >::type opt(optSEXP);
    rcpp_torchoptx_adam_step(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_torchoptx_adam_zero_grad
void rcpp_torchoptx_adam_zero_grad(torchoptx::optim_adam opt);
RcppExport SEXP _torchoptx_rcpp_torchoptx_adam_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_adam >::type opt(optSEXP);
    rcpp_torchoptx_adam_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_torchoptx_adam_step2
void rcpp_torchoptx_adam_step2(torchoptx::optim_adam opt);
RcppExport SEXP _torchoptx_rcpp_torchoptx_adam_step2(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_adam >::type opt(optSEXP);
    rcpp_torchoptx_adam_step2(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_sgd
void rcpp_delete_optim_sgd(void* x);
RcppExport SEXP _torchoptx_rcpp_delete_optim_sgd(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_sgd(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_adam
void rcpp_delete_optim_adam(void* x);
RcppExport SEXP _torchoptx_rcpp_delete_optim_adam(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_adam(x);
    return R_NilValue;
END_RCPP
}
// optim_sgd_new
SEXP optim_sgd_new(torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov);
RcppExport SEXP _torchoptx_optim_sgd_new(SEXP paramsSEXP, SEXP lrSEXP, SEXP momentumSEXP, SEXP dampeningSEXP, SEXP weight_decaySEXP, SEXP nesterovSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type momentum(momentumSEXP);
    Rcpp::traits::input_parameter< double >::type dampening(dampeningSEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< bool >::type nesterov(nesterovSEXP);
    rcpp_result_gen = Rcpp::wrap(optim_sgd_new(params, lr, momentum, dampening, weight_decay, nesterov));
    return rcpp_result_gen;
END_RCPP
}
// optim_sgd_step
void optim_sgd_step(torchoptx::optim_sgd opt);
RcppExport SEXP _torchoptx_optim_sgd_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_sgd >::type opt(optSEXP);
    optim_sgd_step(opt);
    return R_NilValue;
END_RCPP
}
// optim_sgd_zero_grad
void optim_sgd_zero_grad(torchoptx::optim_sgd opt);
RcppExport SEXP _torchoptx_optim_sgd_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_sgd >::type opt(optSEXP);
    optim_sgd_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// optim_adam_new
SEXP optim_adam_new(torch::TensorList params, double lr, double betas0, double betas1, double eps, double weight_decay, bool amsgrad);
RcppExport SEXP _torchoptx_optim_adam_new(SEXP paramsSEXP, SEXP lrSEXP, SEXP betas0SEXP, SEXP betas1SEXP, SEXP epsSEXP, SEXP weight_decaySEXP, SEXP amsgradSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type betas0(betas0SEXP);
    Rcpp::traits::input_parameter< double >::type betas1(betas1SEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< bool >::type amsgrad(amsgradSEXP);
    rcpp_result_gen = Rcpp::wrap(optim_adam_new(params, lr, betas0, betas1, eps, weight_decay, amsgrad));
    return rcpp_result_gen;
END_RCPP
}
// optim_adam_step
void optim_adam_step(torchoptx::optim_adam opt);
RcppExport SEXP _torchoptx_optim_adam_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_adam >::type opt(optSEXP);
    optim_adam_step(opt);
    return R_NilValue;
END_RCPP
}
// optim_adam_zero_grad
void optim_adam_zero_grad(torchoptx::optim_adam opt);
RcppExport SEXP _torchoptx_optim_adam_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_adam >::type opt(optSEXP);
    optim_adam_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// optim_adam_step2
void optim_adam_step2(torchoptx::optim_adam opt);
RcppExport SEXP _torchoptx_optim_adam_step2(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torchoptx::optim_adam >::type opt(optSEXP);
    optim_adam_step2(opt);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_torchoptx_rcpp_torchoptx_sgd", (DL_FUNC) &_torchoptx_rcpp_torchoptx_sgd, 6},
    {"_torchoptx_rcpp_torchoptx_sgd_step", (DL_FUNC) &_torchoptx_rcpp_torchoptx_sgd_step, 1},
    {"_torchoptx_rcpp_torchoptx_sgd_zero_grad", (DL_FUNC) &_torchoptx_rcpp_torchoptx_sgd_zero_grad, 1},
    {"_torchoptx_rcpp_torchoptx_adam", (DL_FUNC) &_torchoptx_rcpp_torchoptx_adam, 7},
    {"_torchoptx_rcpp_torchoptx_adam_step", (DL_FUNC) &_torchoptx_rcpp_torchoptx_adam_step, 1},
    {"_torchoptx_rcpp_torchoptx_adam_zero_grad", (DL_FUNC) &_torchoptx_rcpp_torchoptx_adam_zero_grad, 1},
    {"_torchoptx_rcpp_torchoptx_adam_step2", (DL_FUNC) &_torchoptx_rcpp_torchoptx_adam_step2, 1},
    {"_torchoptx_rcpp_delete_optim_sgd", (DL_FUNC) &_torchoptx_rcpp_delete_optim_sgd, 1},
    {"_torchoptx_rcpp_delete_optim_adam", (DL_FUNC) &_torchoptx_rcpp_delete_optim_adam, 1},
    {"_torchoptx_optim_sgd_new", (DL_FUNC) &_torchoptx_optim_sgd_new, 6},
    {"_torchoptx_optim_sgd_step", (DL_FUNC) &_torchoptx_optim_sgd_step, 1},
    {"_torchoptx_optim_sgd_zero_grad", (DL_FUNC) &_torchoptx_optim_sgd_zero_grad, 1},
    {"_torchoptx_optim_adam_new", (DL_FUNC) &_torchoptx_optim_adam_new, 7},
    {"_torchoptx_optim_adam_step", (DL_FUNC) &_torchoptx_optim_adam_step, 1},
    {"_torchoptx_optim_adam_zero_grad", (DL_FUNC) &_torchoptx_optim_adam_zero_grad, 1},
    {"_torchoptx_optim_adam_step2", (DL_FUNC) &_torchoptx_optim_adam_step2, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_torchoptx(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
