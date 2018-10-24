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

REGISTER_OP("ZeroOut")
  .Input("to_zero: int32")
  .Output("zeroed: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_OP("MinDensity")
  .Input("match_matrix: float")
  .Input("dq_size: int32")
  .Input("location: float")
  .Input("min_density: float")
  .Output("next_location: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input0;
    ::tensorflow::shape_inference::ShapeHandle input1;
    ::tensorflow::shape_inference::ShapeHandle input2;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input0));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input1));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &input2));
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 4));
    return Status::OK();
  });

REGISTER_OP("MinDensityMultiCpu")
  .Input("match_matrix: float")
  .Input("dq_size: int32")
  .Input("location: float")
  .Input("min_density: float")
  .Input("min_jump_offset: int32")
  .Input("use_ratio: bool")
  .Output("next_location: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle input0;
    ::tensorflow::shape_inference::ShapeHandle input1;
    ::tensorflow::shape_inference::ShapeHandle input2;
    ::tensorflow::shape_inference::ShapeHandle input3;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input0));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input1));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &input2));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input3));
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 4));
    return Status::OK();
  });

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

// find the first region that are qualified (denser that min_density)
class MinDensityOp : public OpKernel {
  public:
    explicit MinDensityOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // input
      const Tensor& match_matrix_t = context->input(0);
      const Tensor& dq_size_t = context->input(1);
      const Tensor& location_t = context->input(2);
      const Tensor& min_density_t = context->input(3);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(min_density_t.shape()),
        errors::InvalidArgument("Must be a scalar"));
      auto match_matrix = match_matrix_t.tensor<float, 3>();
      auto dq_size = dq_size_t.tensor<int32, 2>();
      auto location = location_t.tensor<float, 2>();
      auto min_density = min_density_t.scalar<float>()();
      const int64 batch_size = match_matrix_t.dim_size(0);
      // output
      TensorShape output_shape;
      output_shape.AddDim(batch_size);
      output_shape.AddDim(4);
      Tensor* next_location_t = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &next_location_t));
      auto next_location = next_location_t->tensor<float, 2>();
      // find location one by one
      int64 dur_all = 0;
      for (int64 b = 0; b < batch_size; b++) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        const int64 d_len = dq_size(b, 0);
        const int64 q_len = dq_size(b, 1);
        const int64 q_start = (int)floor(location(b, 1));
        const int64 q_offset = (int)floor(location(b, 3));
        const int64 d_start = (int)floor(location(b, 0));
        const int64 d_offset = (int)floor(location(b, 2));
        /*
        std::cout << "batch sample: " << b
                  << ", start ind: " << d_start
                  << ", end ind: " << d_start + d_offset - 1
                  << ", doc len: " << d_len
                  << std::endl;
        */
        OP_REQUIRES(context, q_start < q_len, errors::InvalidArgument("query start ind overflow"));
        OP_REQUIRES(context, q_start + q_offset - 1 < q_len, errors::InvalidArgument("query end ind overflow"));
        OP_REQUIRES(context, d_start < d_len, errors::InvalidArgument("doc start ind overflow"));
        OP_REQUIRES(context, d_start + d_offset - 1 < d_len, errors::InvalidArgument("doc end ind overflow"));
        float* density = (float*)malloc(d_offset * sizeof(float)); // density for the region
        float* density_per = (float*)malloc(d_offset * sizeof(float)); // density for the position
        int* density_offset = (int*)malloc(d_offset * sizeof(int)); // region span
        int64 max_end = -1;
        int64 max_offset = -1;
        for (int64 i = 0; i < d_offset; i++) {
          float cur_density = std::numeric_limits<float>::min();
          for (int64 j = q_start; j < q_start + q_offset; j++) {
            if (match_matrix(b, i + d_start, j) > cur_density) cur_density = match_matrix(b, i + d_start, j);
          }
          density_per[i] = cur_density;
          if (i == 0 || density_offset[i-1] <= 0) {
            // the first position or all the previous ones are not qualified
            if (cur_density < min_density) {
              density[i] = cur_density;
              density_offset[i] = 0;
            } else {
              // find max qualified region
              float new_density = 0;
              for (int64 o = 0; o <= i; o++) {
                new_density = (density_per[i-o] + o * new_density) / (o + 1);
                if (new_density >= min_density) {
                  density[i] = new_density;
                  density_offset[i] = o + 1;
                }
              }
            }
          } else {
            // previous position is qualified
            if (cur_density == min_density) {
              // enlarge previous region by 1
              density_offset[i] = density_offset[i-1] + 1;
              density[i] = (density[i-1] * density_offset[i-1] + cur_density) / density_offset[i];
            } else if (cur_density < min_density) {
              // no larger region
              float new_density = (density[i-1] * density_offset[i-1] + cur_density) / (density_offset[i-1] + 1);
              density[i] = cur_density;
              density_offset[i] = 0;
              for (int64 o = i - density_offset[i-1]; o <= i - 1; o++) {
                if (new_density >= min_density) {
                  density[i] = new_density;
                  density_offset[i] = 1 + i - o;
                  break;
                }
                new_density = (new_density * (1 + i - o) - density_per[o]) / (i - o);
              }
            } else {
              // no smaller region
              float new_density = 0;
              density_offset[i] = density_offset[i-1] + 1;
              density[i] = (density[i-1] * density_offset[i-1] + cur_density) / density_offset[i];
              new_density = density[i];
              for (int64 o = i - density_offset[i-1] - 1; o >= 0; o--) {
                new_density = (density_per[o] + new_density * (i - o)) / (1 + i - o);
                if (new_density >= min_density) {
                  density[i] = new_density;
                  density_offset[i] = 1 + i - o;
                }
              }
            }
          }
          // terminate or continue
          if (density_offset[i] == 0 && max_end != -1) break;
          else if (density_offset[i] > 0 && density_offset[i] > max_offset) {
            max_end = i;
            max_offset = density_offset[i];
          }
        }
        // free
        free(density);
        free(density_per);
        free(density_offset);
        // update location
        if (max_end != -1) {
          next_location(b, 0) = (float)(max_end + d_start - max_offset + 1);
          next_location(b, 1) = location(b, 1);
          next_location(b, 2) = (float)max_offset;
          next_location(b, 3) = location(b, 3);
        } else {
          next_location(b, 0) = (float)(d_start + d_offset); // overflow
          next_location(b, 1) = location(b, 1);
          next_location(b, 2) = 0;
          next_location(b, 3) = location(b, 3);
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto dur = duration_cast<nanoseconds>(t2-t1).count();
        dur_all += dur;
        /*
        std::cout << "batch sample: " << b
                  << ", jump ind: " << next_location(b, 0)
                  << ", jump offset: " << next_location(b, 2)
                  << ", time: " << dur
                  << std::endl;
        */
      }
      //std::cout << "duration (nano): " << dur_all / batch_size << std::endl;
    }
};

// find the first region that are qualified (denser that min_density)
class MinDensityMultiCpuOp : public OpKernel {
  public:
    explicit MinDensityMultiCpuOp(OpKernelConstruction* context) : OpKernel(context) {}

    static void OneTask(int64 start, int64 limit, OpKernelContext* context,
      typename TTypes<float, 3>::ConstTensor match_matrix, typename TTypes<int32, 2>::ConstTensor dq_size,
      typename TTypes<float, 2>::ConstTensor location, typename TTypes<float, 1>::ConstTensor min_density, 
      int32 min_jump_offset, bool use_ratio, typename TTypes<float, 2>::Tensor next_location) {
      for (int64 b = start; b < limit; b++) {
        const int64 d_len = dq_size(b, 0);
        const int64 q_len = dq_size(b, 1);
        const int64 q_start = (int)floor(location(b, 1));
        const int64 q_offset = (int)floor(location(b, 3));
        const int64 d_start = (int)floor(location(b, 0));
        const int64 d_offset = (int)floor(location(b, 2));
        /*
        std::cout << "batch sample: " << b
                  << ", start ind: " << d_start
                  << ", end ind: " << d_start + d_offset - 1
                  << ", doc len: " << d_len
                  << std::endl;
        */
        OP_REQUIRES(context, q_start < q_len, errors::InvalidArgument("query start ind overflow"));
        OP_REQUIRES(context, q_start + q_offset - 1 < q_len, errors::InvalidArgument("query end ind overflow"));
        OP_REQUIRES(context, d_start < d_len, errors::InvalidArgument("doc start ind overflow"));
        OP_REQUIRES(context, d_start + d_offset - 1 < d_len, errors::InvalidArgument("doc end ind overflow"));
        float* density = (float*)malloc(d_offset * sizeof(float)); // density for the region
        float* density_per = (float*)malloc(d_offset * sizeof(float)); // density for the position
        int* density_offset = (int*)malloc(d_offset * sizeof(int)); // region span
        int64 max_end = -1;
        int64 max_offset = -1;
        // find the mean and maximal density
        float mean_density = 0;
        float max_density = std::numeric_limits<float>::min();
        float max_density_track = 0;
        float min_density_value = 0;
        if (use_ratio) {
          for (int64 i = 0; i < d_offset; i++) {
            float cur_density = std::numeric_limits<float>::min();
            // find the maximal similarity among all positions considered in a query
            for (int64 j = q_start; j < q_start + q_offset; j++) {
              if (match_matrix(b, i + d_start, j) > cur_density) cur_density = match_matrix(b, i + d_start, j);
            }
            density_per[i] = cur_density;
            mean_density += cur_density;
            // max_density with length not smaller than min_jump_offset
            //max_density_track += cur_density;
            //if (i >= min_jump_offset) max_density_track -= density_per[i - min_jump_offset];
            //if (i >= min_jump_offset - 1 && max_density_track > max_density) max_density = max_density_track;
            // pointwise max_density
            if (cur_density > max_density) max_density = cur_density;
          }
          mean_density = mean_density / d_offset;
          //max_density = mean_density / min_jump_offset;
          OP_REQUIRES(context, max_density >= mean_density, errors::InvalidArgument("max density smaller than mean density"));
          min_density_value = (max_density - mean_density) * min_density(b) + mean_density;
        } else {
          min_density_value = min_density(b);
        }
        //std::cout << "min_density: " << min_density_value << std::endl;
        for (int64 i = 0; i < d_offset; i++) {
          float cur_density = std::numeric_limits<float>::min();
          // find the maximal similarity among all positions considered in a query
          for (int64 j = q_start; j < q_start + q_offset; j++) {
            if (match_matrix(b, i + d_start, j) > cur_density) cur_density = match_matrix(b, i + d_start, j);
          }
          density_per[i] = cur_density;
          if (i == 0 || density_offset[i-1] <= 0) {
            // the first position or all the previous ones are not qualified
            if (cur_density < min_density_value) {
              density[i] = cur_density;
              density_offset[i] = 0;
            } else {
              // find max qualified region
              float new_density = 0;
              for (int64 o = 0; o <= i; o++) {
                new_density = (density_per[i-o] + o * new_density) / (o + 1);
                if (new_density >= min_density_value) {
                  density[i] = new_density;
                  density_offset[i] = o + 1;
                }
              }
            }
          } else {
            // previous position is qualified
            if (cur_density == min_density_value) {
              // enlarge previous region by 1
              density_offset[i] = density_offset[i-1] + 1;
              density[i] = (density[i-1] * density_offset[i-1] + cur_density) / density_offset[i];
            } else if (cur_density < min_density_value) {
              // no larger region
              float new_density = (density[i-1] * density_offset[i-1] + cur_density) / (density_offset[i-1] + 1);
              density[i] = cur_density;
              density_offset[i] = 0;
              for (int64 o = i - density_offset[i-1]; o <= i - 1; o++) {
                if (new_density >= min_density_value) {
                  density[i] = new_density;
                  density_offset[i] = 1 + i - o;
                  break;
                }
                new_density = (new_density * (1 + i - o) - density_per[o]) / (i - o);
              }
            } else {
              // no smaller region
              float new_density = 0;
              density_offset[i] = density_offset[i-1] + 1;
              density[i] = (density[i-1] * density_offset[i-1] + cur_density) / density_offset[i];
              new_density = density[i];
              for (int64 o = i - density_offset[i-1] - 1; o >= 0; o--) {
                new_density = (density_per[o] + new_density * (i - o)) / (1 + i - o);
                if (new_density >= min_density_value) {
                  density[i] = new_density;
                  density_offset[i] = 1 + i - o;
                }
              }
            }
          }
          // terminate or continue
          // regions smaller than min_jump_offset is not qualified
          if (density_offset[i] == 0 && max_end != -1) break;
          else if (density_offset[i] > 0 && density_offset[i] >= min_jump_offset && density_offset[i] > max_offset) {
            max_end = i;
            max_offset = density_offset[i];
          }
        }
        // free
        free(density);
        free(density_per);
        free(density_offset);
        // update location
        if (max_end != -1) {
          next_location(b, 0) = (float)(max_end + d_start - max_offset + 1);
          next_location(b, 1) = location(b, 1);
          next_location(b, 2) = (float)max_offset;
          next_location(b, 3) = location(b, 3);
        } else {
          next_location(b, 0) = (float)(d_start + d_offset); // overflow
          next_location(b, 1) = location(b, 1);
          next_location(b, 2) = 0;
          next_location(b, 3) = location(b, 3);
        }
        /*
        if (next_location(b, 2) >= 1000) {
          std::cout << "batch sample: " << b
                    << ", jump ind: " << next_location(b, 0)
                    << ", jump offset: " << next_location(b, 2)
                    << std::endl;
        }
        */
      }
    }

    void Compute(OpKernelContext* context) override {
      // input
      const Tensor& match_matrix_t = context->input(0);
      const Tensor& dq_size_t = context->input(1);
      const Tensor& location_t = context->input(2);
      const Tensor& min_density_t = context->input(3);
      const Tensor& min_jump_offset_t = context->input(4);
      const Tensor& use_ratio_t = context->input(5);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(min_jump_offset_t.shape()),
        errors::InvalidArgument("Must be a scalar"));
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(use_ratio_t.shape()),
        errors::InvalidArgument("Must be a scalar"));
      auto match_matrix = match_matrix_t.tensor<float, 3>();
      auto dq_size = dq_size_t.tensor<int32, 2>();
      auto location = location_t.tensor<float, 2>();
      auto min_density = min_density_t.tensor<float, 1>();
      auto min_jump_offset = min_jump_offset_t.scalar<int32>()();
      auto use_ratio = use_ratio_t.scalar<bool>()();
      const int64 batch_size = match_matrix_t.dim_size(0);
      // output
      TensorShape output_shape;
      output_shape.AddDim(batch_size);
      output_shape.AddDim(4);
      Tensor* next_location_t = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &next_location_t));
      auto next_location = next_location_t->tensor<float, 2>();
      // find location using multiple cpu
      auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
      /*
      std::cout << "num threads: " << worker_threads.num_threads
                << "num units: " << batch_size
                << std::endl;
      */
      Shard(1, worker_threads.workers, batch_size, 2500,
        [context, match_matrix, dq_size, location, min_density,  min_jump_offset, use_ratio, next_location]
        (int64 start, int64 limit) {
          MinDensityMultiCpuOp::OneTask(start, limit, context, match_matrix, dq_size, location,
            min_density, min_jump_offset, use_ratio, next_location);
        });
    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
REGISTER_KERNEL_BUILDER(Name("MinDensity").Device(DEVICE_CPU), MinDensityOp);
REGISTER_KERNEL_BUILDER(Name("MinDensityMultiCpu").Device(DEVICE_CPU), MinDensityMultiCpuOp);