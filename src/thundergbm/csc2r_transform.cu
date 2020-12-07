//
// Created by qinbin on 21/8/18.
//

#include "thundergbm/csc2r_transform.h"

void Csc2r::from_csr(float_type* csr_val, int* csr_col_ind, int* csr_row_ptr, int n_instances, int n_column, int nnz){

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    //std::cout<<"nnz:"<<nnz<<std::endl;
    csc_val.resize(nnz);
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(n_column + 1);
    this->nnz = nnz;
    
    cusparseScsr2csc(handle, n_instances, n_column, nnz, csr_val, csr_row_ptr,
                     csr_col_ind, csc_val.device_data(), csc_row_ind.device_data(), csc_col_ptr.device_data(),
                     CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);


}


void Csc2r::get_cut_points_evenly(int nBin, vector<float>& bin_id,
        const vector<float>& min_fea, const vector<float>& max_fea) {
    float* csc_val_host = csc_val.host_data();
    int* csc_row_host = csc_row_ind.host_data();
    int* csc_col_host = csc_col_ptr.host_data();
    for(int cid = 0; cid < csc_col_ptr.size() - 1; cid ++){
        int cstart = csc_col_host[cid];
        int cend = csc_col_host[cid + 1];
        for(int off = cstart; off < cend; off++){

            float val = csc_val_host[off];
            int rid = csc_row_host[off];
//            std::cout<<"rid:"<<rid<<" ";
//            std::cout<<"diff fea:"<<(max_fea[rid] - min_fea[rid])<<" ";
            if((max_fea[rid] - min_fea[rid]) < 1e-5) {
//                std::cout << "only one feature value" << std::endl;
                bin_id[off] = 2.0;
            }
//            if(min_fea[rid] == INFINITY || max_fea[rid] == -INFINITY){
//                std::cout<<"impossible case"<<std::endl;
//                bin_id[off]=0.0;
//            }
            else
                bin_id[off] = 1.0 * ((int) ((val - min_fea[rid]) / (max_fea[rid] - min_fea[rid]) * nBin) + 1);
        }
    }
}


