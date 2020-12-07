#include <thundergbm/util/cub_wrapper.h>
#include "thundergbm/gpu_lsh.h"
#include "thundergbm/util/device_lambda.cuh"



void psdLsh::init(){
    tables.resize(param.n_table);
    //std::cout<<"init n_table:"<<param.n_table<<std::endl;
    //std::cout<<"init n_dimension:"<<param.n_dimension<<std::endl;
//    v_a.resize(param.n_table * param.n_dimension);
    int seed = param.seed;
    std::mt19937 rng(seed==-1?unsigned(std::time(0)):seed);
    //if(seed!=-1)
    //    std::mt19937 rng(seed);    
    //else
    //    std::mt19937 rng(unsigned(std::time(0)));
    //std::mt19937 rng(unsigned(std::time(0))); //random number generator
//    std::mt19937 rng(42);
    std::ofstream outfile;
    outfile.open("unbalance_label.txt", std::ios::out | std::ios::app);
    outfile<<"lsh seed:"<<unsigned(std::time(0))<<std::endl;
    outfile.close();
//    std::cout<<"time 0:"<<std::time(0);
//    std::mt19937 rng(34); // with fixed seed
    std::uniform_real_distribution<float> ur(0, param.r);

    switch (param.p_norm)
    {
        case 1: //CAUCHY
        {

            for(unsigned i = 0; i < param.n_table; i++){
                std::cauchy_distribution<float> cd;
                for(unsigned j = 0; j < param.n_dimension; j++)
                    v_a.push_back(cd(rng));
//                    random_vector.push_back(rng());
                v_b.push_back(ur(rng));
            }
            break;
//            for (std::vector<std::vector<float> >::iterator iter = stableArray.begin(); iter != stableArray.end(); ++iter)
//            {
//                for (unsigned i = 0; i != param.D; ++i)
//                {
//                    iter->push_back(cd(rng));
//                }
//                rndBs.push_back(ur(rng));
//            }
//            return;
        }
        case 2: //GAUSSIAN
        {
            for(unsigned i = 0; i < param.n_table; i++){
                std::normal_distribution<float> nd;
                for(unsigned j = 0; j < param.n_dimension; j++)
                    v_a.push_back(nd(rng));
//                    random_vector.push_back(rng());
                v_b.push_back(ur(rng));
            }
            break;

//            for (std::vector<std::vector<float> >::iterator iter = stableArray.begin(); iter != stableArray.end(); ++iter)
//            {
//                for (unsigned i = 0; i != param.D; ++i)
//                {
//                    iter->push_back(nd(rng));
//                }
//                rndBs.push_back(ur(rng));
//            }
//            return;
        }
        default:
        {
            break;
        }
    }

    a.resize(v_a.size());
    a.copy_from(v_a.data(), v_a.size());
    b.resize(v_b.size());
    b.copy_from(v_b.data(), v_b.size());

    for(int i = 0; i < param.n_table; i++){
//        tables[i].resize(param.n_bucket);
        for(int j = 0; j < param.n_bucket; j++)
            tables[i][j].resize(param.n_comp);
    }
}

void psdLsh::reset(const Parameter &param_)
{
    param = param_;
    init();
}


void psdLsh::hash(int n_instances, int n_features, int nnz, int key_offset,
        SyncArray<float_type> &csr_val, SyncArray<int> &csr_row_ptr, SyncArray<int> &csr_col_ind, SyncArray<int> &hash_values, int cid) {
//    cudaSetDevice(0);

    CHECK(n_features == param.n_dimension);
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    float one(1);
    float zero(0);
    SyncArray<float_type> result(n_instances * param.n_table);
    float *result_device = result.device_data();
    float *b_device = b.device_data();
    cudaDeviceSynchronize();
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_instances, param.n_table, n_features, nnz, &one, descr,
                   csr_val.device_data(), csr_row_ptr.device_data(), csr_col_ind.device_data(), a.device_data(),
                   n_features,
                   &zero, result_device, n_instances);

    cudaDeviceSynchronize();

    int *hash_values_device = hash_values.device_data();

    SyncArray<float> r_gpu(1);
    r_gpu.host_data()[0] = param.r;
    SyncArray<int> n_table_gpu(1);
    n_table_gpu.host_data()[0] = param.n_table;
    cudaDeviceSynchronize();
    float* r_device = r_gpu.device_data();
    int* n_table_device = n_table_gpu.device_data();
    device_loop(result.size(), [=]__device__(int vid){
        result_device[vid] = result_device[vid] + b_device[vid % n_table_device[0]];
        result_device[vid] = result_device[vid] / r_device[0];
        hash_values_device[vid] = __float2int_rd(result_device[vid]);
//        hash_values_device[vid] = (int) (floorf(result_device[vid]));
    });
    cudaDeviceSynchronize();
    int *hash_values_host = hash_values.host_data();
    for (unsigned nid = 0; nid < n_instances; nid++) {
        for (unsigned tid = 0; tid < param.n_table; tid++) {
            hash_values_host[nid * param.n_table + tid] %= param.n_bucket;
//            std::cout<<"hash value"<<hash_values_host[nid * param.n_table + tid]<<std::endl;
            tables[tid][hash_values_host[nid * param.n_table + tid]][cid].push_back(key_offset);
//            if(tables[tid][hash_values_host[nid * param.n_table + tid]].size() == 0)
//                tables[tid][hash_values_host[nid * param.n_table + tid]].resize(param.n_comp);
//            tables[tid][hash_values_host[nid * param.n_table + tid]][cid].push_back(key_offset);
        }
        key_offset ++;
    }
    cudaDeviceSynchronize();
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

//void psdLsh::query(int n_instances, int n_features, int nnz,
//                   SyncArray<float_type> &csr_val, SyncArray<int> &csr_row_ptr, SyncArray<int> &csr_col_ind,
//                   vector<vector<int>>& buckets){
//    CHECK(n_features == param.n_dimension);
//    cusparseHandle_t handle;
//    cusparseMatDescr_t descr;
//    cusparseCreate(&handle);
//    cusparseCreateMatDescr(&descr);
//    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
//    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//    float one(1);
//    float zero(0);
//    SyncArray <float_type> result(n_instances * param.n_table);
//    float *result_device = result.device_data();
//    float *b_device = b.device_data();
//    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_instances, param.n_table, n_features, nnz, &one, descr,
//                   csr_val.device_data(), csr_row_ptr.device_data(), csr_col_ind.device_data(), a.device_data(),
//                   n_features,
//                   &zero, result_device, n_instances);
//
//    SyncArray<int> hash_values(n_instances * param.n_table);
//    int *hash_values_device = hash_values.device_data();
//    int *hash_values_host = hash_values.host_data();
//    device_loop(n_instances * param.n_table, [=]
//    __device__(int vid){
//        result_device[vid] += b_device[vid % param.n_table];
//        result_device[vid] /= param.r;
//        hash_values_device[vid] = (int) (floorf(result_device[vid]));
//    });
//    for (unsigned nid = 0; nid < n_instances; nid++) {
//        for (unsigned tid = 0; tid < param.n_table; tid++) {
//            int bid = hash_values_host[nid * n_instances + tid] % param.n_bucket;
//            buckets[nid].insert(buckets[nid].end(), tables[tid][bid].begin(), tables[tid][bid].end());
//        }
//    }
//}
