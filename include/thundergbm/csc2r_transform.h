//
// Created by qinbin on 21/8/18.
//


#ifndef THUNDERGBM_CSC2R_TRANSFORM_H
#define THUNDERGBM_CSC2R_TRANSFORM_H

#include "thundergbm.h"
#include "syncarray.h"
#include "cusparse.h"
#include "sparse_columns.h"
//change csc 2 csr or csr 2 csc

class Csc2r{
public:
    SyncArray<float_type> csc_val;
    SyncArray<int> csc_row_ind;
    SyncArray<int> csc_col_ptr;
    int nnz;
    //for gpu
    void from_csr(float_type* val, int* csr_col_ind, int* csr_row_ptr, int n_instances, int n_column, int nnz);
    void get_cut_points_evenly(int nBin, vector<float>& bin_id, const vector<float>& min_fea, const vector<float>& max_fea);
//    void init_bin_id_csr(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns, int n_instances);
};

#endif //THUNDERGBM_CSC2R_TRANSFORM_H
