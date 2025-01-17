// Generated by using torchexport::export() -> do not edit by hand
#include <Rcpp.h>
#include <torch.h>
#include "torchoptx_types.h"

torchoptx::optim_sgd rcpp_torchoptx_sgd (torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov);
void rcpp_torchoptx_sgd_step (torchoptx::optim_sgd opt);
void rcpp_torchoptx_sgd_zero_grad (torchoptx::optim_sgd opt);
torchoptx::optim_adam rcpp_torchoptx_adam (torch::TensorList params, double lr, double betas0, double betas1, double eps, double weight_decay, bool amsgrad);
void rcpp_torchoptx_adam_step (torchoptx::optim_adam opt);
void rcpp_torchoptx_adam_zero_grad (torchoptx::optim_adam opt);
void rcpp_torchoptx_adam_step2 (torchoptx::optim_adam opt);
void rcpp_delete_optim_sgd (void* x);
void rcpp_delete_optim_adam (void* x);
