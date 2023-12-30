// modified source https://gist.github.com/marty1885/a939e3cda146e333195bf8f62fde7a95

#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <tuple>
#include <sstream>
#include <iomanip>

#include <cblas.h>
#include <rknn_matmul_api.h>

typedef __fp16 float16;

typedef struct _rknn_matmul_ctx
{
    rknn_context ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_tensor_mem* A;
    rknn_tensor_mem* B;
    rknn_tensor_mem* C;

    float16* a;
    float16* b;
    float* c;

    int32_t M;
    int32_t K;
    int32_t N;
} RKNNMatMulCtx;

RKNNMatMulCtx* make_matmul(int32_t M, int32_t K, int32_t N)
{
    RKNNMatMulCtx* ctx = (RKNNMatMulCtx*)malloc(sizeof(RKNNMatMulCtx));
    memset(ctx, 0, sizeof(RKNNMatMulCtx));

    ctx->info.M             = M;
    ctx->info.K             = K;
    ctx->info.N             = N;
    ctx->info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    ctx->info.AC_layout = 1;
    ctx->info.B_layout   = 1;

    int ret = rknn_matmul_create(&ctx->ctx, &ctx->info, &ctx->io_attr);
    if (ret < 0) {
        printf("rknn_matmul_create fail! ret=%d\n", ret);
        abort();
    }

    // Create A
    ctx->A = rknn_create_mem(ctx->ctx, ctx->io_attr.A.size);
    ctx->B = rknn_create_mem(ctx->ctx, ctx->io_attr.B.size);
    ctx->C = rknn_create_mem(ctx->ctx, ctx->io_attr.C.size);

    ctx->M = M;
    ctx->K = K;
    ctx->N = N;

    ctx->a = (float16*)ctx->A->virt_addr;
    ctx->b = (float16*)ctx->B->virt_addr;
    ctx->c = (float*)ctx->C->virt_addr;

    rknn_matmul_set_io_mem(ctx->ctx, ctx->A, &ctx->io_attr.A);
    rknn_matmul_set_io_mem(ctx->ctx, ctx->B, &ctx->io_attr.B);
    rknn_matmul_set_io_mem(ctx->ctx, ctx->C, &ctx->io_attr.C);

    return ctx;
}

void set_matrix_data(rknn_matmul_ctx* ctx, rknn_tensor_mem* mem, rknn_matmul_tensor_attr* attr, const float* data)
{
    size_t size = mem->size / sizeof(float16);
    float16* ptr = (float16*)mem->virt_addr;
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = (float16)data[i];
    }
    rknn_matmul_set_io_mem(*ctx, mem, attr);
}

void free_matmul(RKNNMatMulCtx* ctx)
{
    rknn_destroy_mem(ctx->ctx, ctx->A);
    rknn_destroy_mem(ctx->ctx, ctx->B);
    rknn_destroy_mem(ctx->ctx, ctx->C);

    rknn_matmul_destroy(ctx->ctx);

    free(ctx);
}

double calculateAverage(const std::vector<float> &vec)
{
    double sum = 0.0;
    for (float value : vec)
    {
        sum += value;
    }
    return sum / vec.size();
}

double calculateStdDev(const std::vector<float> &vec)
{
    double sum = 0.0;

    auto mean = calculateAverage(vec);
    
    for (float value : vec)
    {
        sum += (value - mean) * (value - mean);
    }
    return std::sqrt(sum / vec.size());
}

template <typename T>
std::vector<T> make_random_matrix(size_t M, size_t N) {
  std::vector<T> A(M * N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(0.0, 1.0);
  for (size_t i = 0; i < M * N; ++i) {
    A[i] = dis(gen);
  }
  return A;
}

std::vector<float> matmul_naive(const std::vector<float>& A,
                                const std::vector<float>& B, size_t M, size_t K, size_t N) {
    std::vector<float> C(M * N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return C;
}

std::vector<float> matmul_cblas(const std::vector<float>& A,
                                const std::vector<float>& B, size_t M, size_t K, size_t N) {
  std::vector<float> C(M * N);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
  return C;
}

template <typename Func>
auto benchmark(size_t M, size_t K, size_t N, size_t repeat, Func func) {
  auto A = make_random_matrix<float>(M, K);
  auto B = make_random_matrix<float>(K, N);
  auto start = std::chrono::system_clock::now();

  std::vector<float> times;
  for (size_t i = 0; i < repeat; ++i) {
      auto start = std::chrono::system_clock::now();
      auto C = func(A, B, M, K, N);
      auto end = std::chrono::system_clock::now();
      times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f);
  }
  return std::make_tuple(*std::min_element(times.begin(), times.end()), calculateAverage(times), calculateStdDev(times));
}

auto bench_all(size_t M, size_t K, size_t N, size_t repeat)
{
    std::vector<std::tuple<float, double, double>> ret;
    ret.push_back(benchmark(M, K, N, 3, matmul_naive));
    ret.push_back(benchmark(M, K, N, repeat, matmul_cblas));

    auto rknn_setup = [&](){
        auto ctx = make_matmul(M, K, N);
        return [ctx](const std::vector<float>& A, const std::vector<float>& B, size_t M, size_t K, size_t N) {
            static bool first = true;
            if (first) {
                first = false;
                set_matrix_data(&ctx->ctx, ctx->A, &ctx->io_attr.A, A.data());
                set_matrix_data(&ctx->ctx, ctx->B, &ctx->io_attr.B, B.data());
            }
            rknn_matmul_run(ctx->ctx);
            return std::vector<float>(ctx->c, ctx->c + M * N);
        };
    };

    auto rknn = rknn_setup();
    ret.push_back(benchmark(M, K, N, repeat, rknn));
    return ret;
}



int main()
{
    std::vector<size_t> sizes = {256, 512, 1024, 2048};
    size_t repeat = 20;

    std::cout << "| size | naive, (min, avg, std) | cblas, (min, avg, std) | rknn, (min, avg, std) |" << std::endl;
    std::cout << "| --- | --- | --- | --- |" << std::endl;

    auto tupleToString = [](const std::tuple<float, double, double> &t)
    {
        std::stringstream ss;
        ss <<  std::setprecision(2);
        ss << "(" << std::get<0>(t) << ", " << std::get<1>(t) << ", " << std::get<2>(t) << ")";
        return ss.str();
    };
    for (auto size : sizes) {
        auto ret = bench_all(size, size, size, repeat);
        std::cout << "|" << size << "|" << tupleToString(ret[0]) << "|" << tupleToString(ret[1])
                  << "|" << tupleToString(ret[2]) << "|" << std::endl;
    }

    return 0;
}