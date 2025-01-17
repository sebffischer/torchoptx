// Generated by using torchexport::export() -> do not edit by hand
#include "torchoptx/torchoptx.h"
#include <lantern/types.h>
#include "torchoptx/torchoptx_types.h"
void * p_torchoptx_last_error = NULL;

TORCHOPTX_API void* torchoptx_last_error()
{
  return p_torchoptx_last_error;
}

TORCHOPTX_API void torchoptx_last_error_clear()
{
  p_torchoptx_last_error = NULL;
}

optim_sgd torchoptx_sgd (torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov);
TORCHOPTX_API void* _torchoptx_sgd (void* params, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  try {
    return  make_raw::SGD(torchoptx_sgd(from_raw::TensorList(params), lr, momentum, dampening, weight_decay, nesterov));
  } TORCHOPTX_HANDLE_EXCEPTION
  return (void*) NULL;
}
void torchoptx_sgd_step (optim_sgd opt);
TORCHOPTX_API void _torchoptx_sgd_step (void* opt) {
  try {
     (torchoptx_sgd_step(from_raw::SGD(opt)));
  } TORCHOPTX_HANDLE_EXCEPTION
  
}
void torchoptx_sgd_zero_grad (optim_sgd opt);
TORCHOPTX_API void _torchoptx_sgd_zero_grad (void* opt) {
  try {
     (torchoptx_sgd_zero_grad(from_raw::SGD(opt)));
  } TORCHOPTX_HANDLE_EXCEPTION
  
}
optim_adam torchoptx_adam (torch::TensorList params, double lr, double betas0, double betas1, double eps, double weight_decay, bool amsgrad);
TORCHOPTX_API void* _torchoptx_adam (void* params, double lr, double betas0, double betas1, double eps, double weight_decay, bool amsgrad) {
  try {
    return  make_raw::Adam(torchoptx_adam(from_raw::TensorList(params), lr, betas0, betas1, eps, weight_decay, amsgrad));
  } TORCHOPTX_HANDLE_EXCEPTION
  return (void*) NULL;
}
void torchoptx_adam_step (optim_adam opt);
TORCHOPTX_API void _torchoptx_adam_step (void* opt) {
  try {
     (torchoptx_adam_step(from_raw::Adam(opt)));
  } TORCHOPTX_HANDLE_EXCEPTION
  
}
void torchoptx_adam_zero_grad (optim_adam opt);
TORCHOPTX_API void _torchoptx_adam_zero_grad (void* opt) {
  try {
     (torchoptx_adam_zero_grad(from_raw::Adam(opt)));
  } TORCHOPTX_HANDLE_EXCEPTION
  
}
void torchoptx_adam_step2 (optim_adam opt);
TORCHOPTX_API void _torchoptx_adam_step2 (void* opt) {
  try {
     (torchoptx_adam_step2(from_raw::Adam(opt)));
  } TORCHOPTX_HANDLE_EXCEPTION
  
}
void delete_optim_sgd (void* x);
TORCHOPTX_API void _delete_optim_sgd (void* x) {
  try {
     (delete_optim_sgd(x));
  } TORCHOPTX_HANDLE_EXCEPTION
  
}
void delete_optim_adam (void* x);
TORCHOPTX_API void _delete_optim_adam (void* x) {
  try {
     (delete_optim_adam(x));
  } TORCHOPTX_HANDLE_EXCEPTION
  
}
