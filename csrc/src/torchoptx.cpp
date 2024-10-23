#include <torch/torch.h>
#define LANTERN_TYPES_IMPL // Should be defined only in a single file.
#include <lantern/types.h>
#include <iostream>
#include <vector>
#include "torchoptx/torchoptx.h"
#include "torchoptx/torchoptx_types.h"
#include <torch/script.h>  // One-stop header.

using namespace torch::jit;

// [[torch::export(register_types=c("optim_sgd", "SGD", "void*", "torchoptx::optim_sgd"))]]
optim_sgd torchoptx_sgd(torch::TensorList params, double lr, double momentum, double dampening,
                        double weight_decay, bool nesterov) {

  auto options = torch::optim::SGDOptions(lr)
    .momentum(momentum)
    .dampening(dampening)
    .weight_decay(weight_decay)
    .nesterov(nesterov);
 return new torch::optim::SGD(params.vec(), options);
}

// [[torch::export]]
void torchoptx_sgd_step(optim_sgd opt) {
  opt->step();
}

// [[torch::export]]
void torchoptx_sgd_zero_grad(optim_sgd opt) {
  opt->zero_grad();
}

// [[torch::export(register_types=c("optim_adam", "Adam", "void*", "torchoptx::optim_adam"))]]
optim_adam torchoptx_adam(torch::TensorList params, double lr, double betas0, double betas1,
                          double eps, double weight_decay, bool amsgrad) {
  auto options = torch::optim::AdamOptions()
    .lr(lr)
    .betas({betas0, betas1})
    .eps(eps)
    .weight_decay(weight_decay)
    .amsgrad(amsgrad);
  return new torch::optim::Adam(params.vec(), options);
}

// [[torch::export]]
void torchoptx_adam_step(optim_adam opt) {
  opt->step();
}

// [[torch::export]]
void torchoptx_adam_zero_grad(optim_adam opt) {
  opt->zero_grad();
}

// [[torch::export]]
void torchoptx_adam_step2(optim_adam opt) {
  opt->step();
}



//void torchoptx_call_traced_fn(const torch::jit::GraphFunction* fn) {
//  auto x = torch::randn({1, 1});
//}

//void* torchoptx_call_traced_fn2(void* fn, void* inputs) {
  //auto fn_ = reinterpret_cast<GraphFunction*>(fn);
  //Stack inputs_ = *reinterpret_cast<Stack*>(inputs);

  //auto outputs = torch::jit::Stack();
  //auto out = (*fn_)(inputs_);
  //outputs.push_back(out);

  //return make_ptr<torch::jit::Stack>(outputs);
//}
