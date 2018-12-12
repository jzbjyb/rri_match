#include "math.h"
#include "limits"
#include "iostream"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include "chrono"

using namespace tensorflow;
using namespace std::chrono;

REGISTER_OP("DynamicSplitMultiCpu")
  .Attr("T: {float, int32}")
  .Input("x: T")
  .Input("start: int32")
  .Input("offset: int32")
  .Output("split_x: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input0;
    ::tensorflow::shape_inference::ShapeHandle input1;
    ::tensorflow::shape_inference::ShapeHandle input2;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input0));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input1));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input2));
    return Status::OK();
  });

REGISTER_OP("DynamicSplitContinuousMultiCpu")
  .Attr("T: {float, int32}")
  .Input("x: T")
  .Input("offset: int32")
  .Output("split_x: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input0;
    ::tensorflow::shape_inference::ShapeHandle input1;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input0));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input1));
    return Status::OK();
  });

template <typename T>
void OneTask(int64 task_start, int64 task_limit, OpKernelContext* context,
  typename TTypes<T, 2>::ConstTensor x, int* start, 
  typename TTypes<int32, 1>::ConstTensor offset, int32 max_offset, int32 x_size, 
  typename TTypes<T, 3>::Tensor split_x) {
  // for each split
  for (int64 b = task_start; b < task_limit; b++) {
    int32 st = start[b];
    int32 of = offset(b);
    int32 end = std::min(st + of, x_size);
    for (int64 i = st; i < st + max_offset; i++)
      for (int64 j = 0; j < x.dimension(1); j++) {
        if (i < end) split_x(b, i - st, j) = x(i, j);
        else split_x(b, i - st, j) = 0;
      }
  }
};

template <typename T>
void CommonCompute(OpKernelContext* context, const Tensor x_t, int* start, 
  typename TTypes<int32, 1>::ConstTensor offset) {
  // input
  const int64 x_size = x_t.dim_size(0);
  const int64 split_num = offset.dimension(0);
  // find the max offset
  int64 max_offset = -1;
  for (int64 i = 0; i < split_num; i++)
    if (offset(i) > max_offset) max_offset = offset(i);
  // shaping
  int64 inner_size = 1;
  TensorShape output_shape;
  output_shape.AddDim(split_num);
  output_shape.AddDim(max_offset);
  for (int i = 1; i < x_t.dims(); i++) {
    output_shape.AddDim(x_t.dim_size(i));
    inner_size *= x_t.dim_size(i);
  }
  // generate output
  Tensor* split_x_t = NULL;
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &split_x_t));
  // flatten x
  auto x = x_t.shaped<T, 2>({x_size, inner_size});
  auto split_x = split_x_t->shaped<T, 3>({split_num, max_offset, inner_size});
  // use multiple cpu
  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
  int64 cost_per_unit = max_offset * inner_size * 3;
  Shard(worker_threads.num_threads, worker_threads.workers, split_num, cost_per_unit,
    [context, x, start, offset, max_offset, x_size, split_x]
    (int64 task_start, int64 task_limit) {
      OneTask<T>(task_start, task_limit, context, 
        x, start, offset, max_offset, x_size, split_x);
    });
};

template <typename T>
class DynamicSplitMultiCpuOp : public OpKernel {
  public:
    explicit DynamicSplitMultiCpuOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // input
      const Tensor& x_t = context->input(0);
      const Tensor& start_t = context->input(1);
      const Tensor& offset_t = context->input(2);
      auto start = start_t.tensor<int32, 1>();
      auto offset = offset_t.tensor<int32, 1>();
      // create start
      int* start_array = (int*)malloc(start_t.dim_size(0) * sizeof(int));
      // copy start
      for (int64 i = 0; i < start_t.dim_size(0); i++)
        start_array[i] = start(i);
      CommonCompute<T>(context, x_t, start_array, offset);
    }
};

template <typename T>
class DynamicSplitContinuousMultiCpuOp : public OpKernel {
  public:
    explicit DynamicSplitContinuousMultiCpuOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // input
      const Tensor& x_t = context->input(0);
      const Tensor& offset_t = context->input(1);
      auto offset = offset_t.tensor<int32, 1>();
      // create start
      int* start_array = (int*)malloc(offset_t.dim_size(0) * sizeof(int));
      // calculate start
      int64 acc = 0;
      for (int64 i = 0; i < offset_t.dim_size(0); i++) {
        start_array[i] = acc;
        acc += offset(i);
      }
      CommonCompute<T>(context, x_t, start_array, offset);
    }
};

REGISTER_KERNEL_BUILDER(
  Name("DynamicSplitMultiCpu")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("T"),
  DynamicSplitMultiCpuOp<int32>);
REGISTER_KERNEL_BUILDER(
  Name("DynamicSplitMultiCpu")
  .Device(DEVICE_CPU)
  .TypeConstraint<float>("T"),
  DynamicSplitMultiCpuOp<float>);

REGISTER_KERNEL_BUILDER(
  Name("DynamicSplitContinuousMultiCpu")
  .Device(DEVICE_CPU)
  .TypeConstraint<int32>("T"),
  DynamicSplitContinuousMultiCpuOp<int32>);
REGISTER_KERNEL_BUILDER(
  Name("DynamicSplitContinuousMultiCpu")
  .Device(DEVICE_CPU)
  .TypeConstraint<float>("T"),
  DynamicSplitContinuousMultiCpuOp<float>);
