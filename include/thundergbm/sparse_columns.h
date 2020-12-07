//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_SPARSE_COLUMNS_H
#define THUNDERGBM_SPARSE_COLUMNS_H

#include "thundergbm.h"
#include "syncarray.h"
#include "dataset.h"
#include "cusparse.h"

class SparseColumns {//one feature corresponding to one column
public:
    SyncArray<float_type> csc_val;
    SyncArray<int> csc_row_ind;
    SyncArray<int> csc_col_ptr;
    SyncArray<float_type> csc_bin_id;
    int n_column;
    int column_offset;
    int nnz;

    void from_dataset(const DataSet &dataSet);
    void from_dataset_csr(const DataSet &dataset);

    void to_multi_devices(vector<std::shared_ptr<SparseColumns>> &) const;
//    void get_cut_points_evenly(int nBin, vector<int>& bin_id, const vector<float>& min_fea, const vector<float>& max_fea);
};
#endif //THUNDERGBM_SPARSE_COLUMNS_H
