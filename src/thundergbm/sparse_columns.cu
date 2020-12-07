//
// Created by shijiashuai on 5/7/18.
//
#include <thundergbm/util/cub_wrapper.h>
#include "thundergbm/sparse_columns.h"
#include "thundergbm/util/device_lambda.cuh"

void SparseColumns::from_dataset(const DataSet &dataset) {
    LOG(TRACE) << "constructing sparse columns";
    n_column = dataset.n_features();
    vector<float_type> csc_val_vec;
    vector<int> csc_row_ind_vec;
    vector<int> csc_col_ptr_vec;
    csc_col_ptr_vec.push_back(0);
    for (int i = 0; i < n_column; i++) {
        csc_val_vec.insert(csc_val_vec.end(), dataset.features[i].begin(), dataset.features[i].end());
        csc_row_ind_vec.insert(csc_row_ind_vec.end(), dataset.line_num[i].begin(), dataset.line_num[i].end());
        csc_col_ptr_vec.push_back(csc_col_ptr_vec.back() + dataset.features[i].size());
    }
    nnz = csc_val_vec.size();
    csc_val.resize(csc_val_vec.size());
    memcpy(csc_val.host_data(), csc_val_vec.data(), sizeof(float_type) * csc_val_vec.size());
    csc_row_ind.resize(csc_row_ind_vec.size());
    memcpy(csc_row_ind.host_data(), csc_row_ind_vec.data(), sizeof(int) * csc_row_ind_vec.size());
    csc_col_ptr.resize(csc_col_ptr_vec.size());
    memcpy(csc_col_ptr.host_data(), csc_col_ptr_vec.data(), sizeof(int) * csc_col_ptr_vec.size());
    cudaDeviceSynchronize();// ?
}

void SparseColumns::from_dataset_csr(const DataSet &dataset) {
    LOG(INFO) << "constructing sparse columns";
    n_column = dataset.n_features();
    size_t n_instances = dataset.n_instances();
    const DataSet::node2d &instances = dataset.instances();

    /**
     * construct csr matrix, then convert to csc matrix and sort columns by feature values
     */
    vector<float_type> csr_val;
    vector<int> csr_col_ind;//index of each value of all the instances
    vector<int> csr_row_ptr(1, 0);//the start positions of the instances

    LOG(INFO) << "converting libsvm sparse rows to csr matrix";
    for (const auto &ins : instances) {//convert libsvm format to csr format
        for (const auto &j : ins) {
            csr_val.push_back(j.value);
            csr_col_ind.push_back(j.index - 1);//libSVM data format is one-based, convert to zero-based
        }
        CHECK_LE(csr_row_ptr.back() + ins.size(), INT_MAX);
        csr_row_ptr.push_back(csr_row_ptr.back() + ins.size());
    }

    nnz = csr_val.size();//number of nonzer
    LOG(INFO)
            << string_format("dataset density = %.2f%% (%d feature values, ave=%d/instance, %d/feature)",
                             (float) nnz / n_instances / n_column * 100,
                             nnz, nnz / n_instances, nnz / n_column);

    LOG(INFO) << "copy csr matrix to GPU";
    //three arrays (on GPU/CPU) for csr representation
    SyncArray<float_type> val;
    SyncArray<int> col_ind;
    SyncArray<int> row_ptr;
    val.resize(csr_val.size());
    col_ind.resize(csr_col_ind.size());
    row_ptr.resize(csr_row_ptr.size());

    //copy data to the three arrays
    val.copy_from(csr_val.data(), val.size());
    col_ind.copy_from(csr_col_ind.data(), col_ind.size());
    row_ptr.copy_from(csr_row_ptr.data(), row_ptr.size());

    LOG(INFO) << "converting csr matrix to csc matrix";
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    csc_val.resize(nnz);
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(n_column + 1);

    cusparseScsr2csc(handle, n_instances, n_column, nnz, val.device_data(), row_ptr.device_data(),
                     col_ind.device_data(), csc_val.device_data(), csc_row_ind.device_data(), csc_col_ptr.device_data(),
                     CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}


void SparseColumns::to_multi_devices(vector<std::shared_ptr<SparseColumns>> &v_columns) const{
    //devide data into multiple devices
    int cur_device_id;
    cudaGetDevice(&cur_device_id);
    int n_device = v_columns.size();
    int ave_n_columns = n_column / n_device;
    DO_ON_MULTI_DEVICES(n_device, [&](int device_id) {
        SparseColumns &columns = *v_columns[device_id];
        const int *csc_col_ptr_data = csc_col_ptr.host_data();
        int first_col_id = device_id * ave_n_columns;
        int n_column_sub = (device_id < n_device - 1) ? ave_n_columns : n_column - first_col_id;
        n_column_sub = (n_device == 1) ? ave_n_columns : n_column_sub;
        int first_col_start = csc_col_ptr_data[first_col_id];
        int nnz_sub = (device_id < n_device - 1) ?
                      (csc_col_ptr_data[(device_id + 1) * ave_n_columns] - first_col_start) : (nnz -
                                                                                               first_col_start);
        nnz_sub = (n_device == 1) ? nnz : nnz_sub;

        columns.column_offset = first_col_id;
        columns.nnz = nnz_sub;
        columns.n_column = n_column_sub;
        columns.csc_val.resize(nnz_sub);
        columns.csc_row_ind.resize(nnz_sub);
        columns.csc_col_ptr.resize(n_column_sub + 1);

        columns.csc_val.copy_from(csc_val.host_data() + first_col_start, nnz_sub);
        columns.csc_row_ind.copy_from(csc_row_ind.host_data() + first_col_start, nnz_sub);
        columns.csc_col_ptr.copy_from(csc_col_ptr.host_data() + first_col_id, n_column_sub + 1);

        int *csc_col_ptr_2d_data = columns.csc_col_ptr.device_data();


        //correct segment start positions
        device_loop(n_column_sub + 1, [=] __device__(int col_id) {
            csc_col_ptr_2d_data[col_id] = csc_col_ptr_2d_data[col_id] - first_col_start;
        });
        LOG(TRACE) << "sorting feature values (multi-device)";
        cub_seg_sort_by_key(columns.csc_val, columns.csc_row_ind, columns.csc_col_ptr, false);

    });
    LOG(TRACE) << "sorting finished";
}

//void SparseColumns::get_cut_points_evenly(int nBin, int max_dimension, vector<int>& bin_id,
//        const vector<float>& min_fea, const vector<float>& max_fea) {
//    float* csc_val_host = csc_val.host_data();
//    int* csc_row_host = csc_row_ind.host_data();
//    int* csc_col_host = csc_col_ptr.host_data();
//    for(int cid = 0; cid < csc_col_ptr.size() - 1; cid ++){
//        cstart = csc_col_host[cid];
//        cend = csc_col_host[cid + 1];
//        for(int off = cstart; off < cend; off++){
//            float val = csc_val_host[off];
//            bin_id[off] = (int) ((val - min_fea[cid]) / (max_fea[cid] - min_fea[cid]) * nBin);
//        }
//    }
//}
