// Generated by using torchexport::export() -> do not edit by hand
#include "exports.h"
#define TORCHOPTX_HEADERS_ONLY
#include <torchoptx/torchoptx.h>

// [[Rcpp::export]]
torchoptx::optim_sgd rcpp_torchoptx_sgd (torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  return  torchoptx_sgd(params.get(), lr, momentum, dampening, weight_decay, nesterov);
}
// [[Rcpp::export]]
void rcpp_torchoptx_sgd_step (torchoptx::optim_sgd opt) {
   torchoptx_sgd_step(opt.get());
}
// [[Rcpp::export]]
void rcpp_torchoptx_sgd_zero_grad (torchoptx::optim_sgd opt) {
   torchoptx_sgd_zero_grad(opt.get());
}
// [[Rcpp::export]]
torchoptx::optim_adam rcpp_torchoptx_adam (torch::TensorList params, double lr, double betas0, double betas1, double eps, double weight_decay, bool amsgrad) {
  return  torchoptx_adam(params.get(), lr, betas0, betas1, eps, weight_decay, amsgrad);
}
// [[Rcpp::export]]
void rcpp_torchoptx_adam_step (torchoptx::optim_adam opt) {
   torchoptx_adam_step(opt.get());
}
// [[Rcpp::export]]
void rcpp_torchoptx_adam_zero_grad (torchoptx::optim_adam opt) {
   torchoptx_adam_zero_grad(opt.get());
}
// [[Rcpp::export]]
void rcpp_torchoptx_adam_step2 (torchoptx::optim_adam opt) {
   torchoptx_adam_step2(opt.get());
}
// [[Rcpp::export]]
void rcpp_delete_optim_sgd (void* x) {
   delete_optim_sgd(x);
}
// [[Rcpp::export]]
void rcpp_delete_optim_adam (void* x) {
   delete_optim_adam(x);
}
