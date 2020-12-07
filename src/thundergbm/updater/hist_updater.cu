#include "thundergbm/updater/hist_updater.h"
#include <thundergbm/util/cub_wrapper.h>
void HistUpdater::insBundle(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats){
//    auto stats_ghpair = stats.gh_pair.host_data();
//    InsStat sstats;
//    int n_instances = stats.n_instances;
////    float g_abs[n_instances];
////    for(int i = 0; i < n_instances; i++){
////        g_abs[i] = fabs(stats_ghpair[i].g);
////    }
////    std::sort(g_abs, g_abs + n_instances);
////    std::cout<<"min g: "<<g_abs[0]<<" max g:"<<g_abs[n_instances - 1]<<" median:"<<g_abs[n_instances/2]<<std::endl;
//    float min = fabs(stats_ghpair[0].g);
//    float max = fabs(stats_ghpair[0].g);
//    for(int i = 0; i < stats.gh_pair.size(); i++){
//        float g_abs = fabs(stats_ghpair[i].g);
//        if(g_abs < min)
//            min = g_abs;
//        if(g_abs > max)
//            max = g_abs;
//    }
//    float pro = 0.01;
//    float threshold = pro * max;
////    vector<insBundle> ibs;
////    for(int i = 0; i < n_instances; i++){
////        for(int i = 0; i < ibs.size(); i++){
////
////        }
////    }

}

void HistUpdater::init_bin_id(const vector<std::shared_ptr<SparseColumns>> &v_columns){
    bin_id.clear();
    bin_id.resize(n_devices);
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        LOG(TRACE) << string_format("get bin ids on device %d", device_id);
        get_bin_ids(*v_columns[device_id]);
    });
    LOG(INFO)<<"after get bin ids";
    int bin_size = (*bin_id[0]).size();
    auto bin_id_ptr = (*bin_id[0]).device_data();
    (*v_columns[0]).csc_bin_id.resize(bin_size);
    auto csc_bin_id_ptr = (*v_columns[0]).csc_bin_id.device_data();
    device_loop(bin_size, [=]__device__(int bid){
        csc_bin_id_ptr[bid] = (float_type)bin_id_ptr[bid];
    });
}

void HistUpdater::init_bin_id_outside(const vector<std::shared_ptr<SparseColumns>> &v_columns, SyncArray<int>& bin_id){
    using namespace thrust;
    int cur_device;
    cudaGetDevice(&cur_device);
    int n_column = (*v_columns[0]).n_column;
    int nnz = (*v_columns[0]).nnz;
    auto cut_row_ptr = v_cut[cur_device].cut_row_ptr.device_data();
    auto cut_points_ptr = v_cut[cur_device].cut_points_val.device_data();
    auto csc_val_data = (*v_columns[0]).csc_val.device_data();
    auto csc_col_data = (*v_columns[0]).csc_col_ptr.device_data();
    bin_id.resize(nnz);
    auto bin_id_ptr = bin_id.device_data();
//    bin_id[cur_device].reset(new SyncArray<int>(nnz));
//    auto bin_id_ptr = (*bin_id[cur_device]).device_data();
    //std::cout<<"get bin id before loop"<<std::endl;
    device_loop(n_column, [=]__device__(int cid){
        auto cutbegin = cut_points_ptr + cut_row_ptr[cid];
        auto cutend = cut_points_ptr + cut_row_ptr[cid + 1];
        auto valbeign = csc_val_data + csc_col_data[cid];
        auto valend = csc_val_data + csc_col_data[cid + 1];
        lower_bound(cuda::par, cutbegin, cutend, valbeign, valend,
                    bin_id_ptr + csc_col_data[cid], thrust::greater<float_type>());
//        for_each(cuda::par, bin_id_ptr + csc_col_data[cid],
//                 bin_id_ptr + csc_col_data[cid + 1], thrust::placeholders::_1 += cut_row_ptr[cid]);
    });
    return;
}

void HistUpdater::init_bin_id_unsort(SparseColumns& unsort_columns, SyncArray<int>& bin_id){
    using namespace thrust;
    int cur_device;
    cudaGetDevice(&cur_device);
    int n_column = unsort_columns.n_column;
    int nnz = unsort_columns.nnz;
    auto cut_row_ptr = v_cut[cur_device].cut_row_ptr.device_data();
    auto cut_points_ptr = v_cut[cur_device].cut_points_val.device_data();
    auto csc_val_data = unsort_columns.csc_val.device_data();
    auto csc_col_data = unsort_columns.csc_col_ptr.device_data();
    bin_id.resize(nnz);
    auto bin_id_ptr = bin_id.device_data();
    //std::cout<<"get bin id before loop"<<std::endl;
    device_loop(n_column, [=]__device__(int cid){
        auto cutbegin = cut_points_ptr + cut_row_ptr[cid];
        auto cutend = cut_points_ptr + cut_row_ptr[cid + 1];
        auto valbeign = csc_val_data + csc_col_data[cid];
        auto valend = csc_val_data + csc_col_data[cid + 1];
        lower_bound(cuda::par, cutbegin, cutend, valbeign, valend,
                    bin_id_ptr + csc_col_data[cid], thrust::greater<float_type>());
//        for_each(cuda::par, bin_id_ptr + csc_col_data[cid],
//                 bin_id_ptr + csc_col_data[cid + 1], thrust::placeholders::_1 += cut_row_ptr[cid]);
    });
}
void HistUpdater::copy_bin_id(const vector<std::shared_ptr<SparseColumns>> &v_columns, SyncArray<int>& out_bin_id){
    int cur_device = 0;
    bin_id.clear();
    bin_id.resize(n_devices);
    bin_id[cur_device].reset(new SyncArray<int>(out_bin_id.size()));
    auto bin_id_ptr = (*bin_id[cur_device]).device_data();
    auto out_bin_ptr = out_bin_id.device_data();
    int bin_size = out_bin_id.size();
    device_loop(bin_size, [=]__device__(int bid){
        bin_id_ptr[bid] = out_bin_ptr[bid];
    });
    (*v_columns[0]).csc_bin_id.resize(bin_size);
    auto csc_bin_id_ptr = (*v_columns[0]).csc_bin_id.device_data();
    device_loop(bin_size, [=]__device__(int bid){
        csc_bin_id_ptr[bid] = (float_type)bin_id_ptr[bid];
    });
    return;
}

void HistUpdater::init_cut(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats, int n_instances,
        SparseColumns& unsort_columns){
    LOG(TRACE)<<"init cut";
    //std::cout<<"n_devices:"<<n_devices<<std::endl;
    if(!do_cut) {
        v_cut.clear();
        v_cut.resize(n_devices);
        LOG(INFO)<<"start init cut";
    {
        TIMED_SCOPE(timerObj, "get cut points");
        for (int i = 0; i < n_devices; i++)
            v_cut[i].get_cut_points(*v_columns[i], stats, max_num_bin, n_instances, i);
    }
    LOG(INFO)<<"cut points:"<<v_cut[0].cut_points_val;
    LOG(INFO)<<"after get cut points";
    {
        TIMED_SCOPE(timerObj, "get bin ids");
        bin_id.clear();
        bin_id.resize(n_devices);
        DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
            LOG(TRACE) << string_format("get bin ids on device %d", device_id);
//            if(use_similar_bundle)
//                get_bin_ids(unsort_columns);
//            else
                get_bin_ids(*v_columns[device_id]);
        });
    }
    LOG(INFO)<<"after get bin ids";
        int bin_size = (*bin_id[0]).size();
        auto bin_id_ptr = (*bin_id[0]).device_data();
        (*v_columns[0]).csc_bin_id.resize(bin_size);
        auto csc_bin_id_ptr = (*v_columns[0]).csc_bin_id.device_data();
        device_loop(bin_size, [=]__device__(int bid){
            csc_bin_id_ptr[bid] = (float_type)bin_id_ptr[bid];
        });
        LOG(INFO)<<"bin id"<<(*bin_id[0]);


        auto csc_bin_id_host = (*v_columns[0]).csc_bin_id.host_data();
        auto csc_col_host = (*v_columns[0]).csc_col_ptr.host_data();
        std::cout<<"bin_id 0:"<<csc_bin_id_host[csc_col_host[1] - 1]<<std::endl;
        std::cout<<"bin_id 1:"<<csc_bin_id_host[csc_col_host[2] - 1]<<std::endl;
//        (*v_columns[0]).csc_val.resize(0);
//        if(!use_similar_bundle){
//            cub_seg_sort_by_key((*v_columns[0]).csc_row_ind, (*bin_id[0]), (*v_columns[0]).csc_col_ptr, true);
//        }
//        cudaDeviceSynchronize();

//        auto csc_col_host = (*v_columns[0]).csc_col_ptr.host_data();
//        auto bin_id_val_host = (*bin_id[0]).host_data();
//        std::cout<<"bin id fifth col:"<<std::endl;
//        for(int i = csc_col_host[4]; i < csc_col_host[5]; i++){
//            std::cout<<bin_id_val_host[i]<<"\t";
//        }
//        std::cout<<std::endl;
    }
    do_cut = 1;


}

void HistUpdater::get_bin_ids(const SparseColumns &columns){
    using namespace thrust;
    int cur_device;
    cudaGetDevice(&cur_device);
    int n_column = columns.n_column;
    int nnz = columns.nnz;
    auto cut_row_ptr = v_cut[cur_device].cut_row_ptr.device_data();
    auto cut_points_ptr = v_cut[cur_device].cut_points_val.device_data();
    auto csc_val_data = columns.csc_val.device_data();
    auto csc_col_data = columns.csc_col_ptr.device_data();
    bin_id[cur_device].reset(new SyncArray<int>(nnz));
    auto bin_id_ptr = (*bin_id[cur_device]).device_data();
    //std::cout<<"get bin id before loop"<<std::endl;
    device_loop(n_column, [=]__device__(int cid){
        auto cutbegin = cut_points_ptr + cut_row_ptr[cid];
        auto cutend = cut_points_ptr + cut_row_ptr[cid + 1];
        auto valbeign = csc_val_data + csc_col_data[cid];
        auto valend = csc_val_data + csc_col_data[cid + 1];
        lower_bound(cuda::par, cutbegin, cutend, valbeign, valend,
                    bin_id_ptr + csc_col_data[cid], thrust::greater<float_type>());
//        for_each(cuda::par, bin_id_ptr + csc_col_data[cid],
//                 bin_id_ptr + csc_col_data[cid + 1], thrust::placeholders::_1 += cut_row_ptr[cid]);
    });
}

//only for single device now
//non-continuous memory operations
//can be done in dataset::load
//void HistUpdater::similar_ins_bundle(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats, int n_instances){
//    std::cout<<"begin"<<std::endl;
//    using namespace thrust;
//    SparseColumns& columns = *v_columns[0];
//    int cur_device;
//    cudaGetDevice(&cur_device);
//    int n_feature = columns.n_column;
//    std::cout<<"n_feature:"<<n_feature<<std::endl;
//    SyncArray<int> dense_binids(n_feature * n_instances);
//    int* dense_id_host = dense_binids.host_data();
//    float_type* val_ptr = columns.csc_val.host_data();
//    int* row_ptr = columns.csc_row_ind.host_data();
//    int* col_ptr = columns.csc_col_ptr.host_data();
//    int last_iid = 0;
//    int* bin_id_data = (*bin_id[cur_device]).host_data();
//#pragma omp parallel for
//    for(int i = 0; i < columns.csc_col_ptr.size() - 1; i++){
//        last_iid = 0;
//        if(col_ptr[i] == col_ptr[i+1]){
//            for(int k = 0; k < n_instances; k++){
//                dense_id_host[k * n_feature + i] = -1;
//            }
//        }
//        for(int j = col_ptr[i]; j < col_ptr[i + 1]; j++){
//            int iid = row_ptr[j];
//            //float_type val = val_ptr[j];
////            for(int k = last_iid; k < iid; k++){
////                dense_id_host[k * n_feature + i] = -1;
////            }
//            dense_id_host[iid * n_feature + i] = bin_id_data[j];
//            last_iid = iid + 1;
//        }
//    }
//    //int* dense_id_host = dense_binids.host_data();
//    vector<bool> in_bundle(n_instances);
//    vector<int> to_del;
//    std::cout<<"1"<<std::endl;
//    for(int i = 0; i < n_instances - 1; i++){
//        if(in_bundle[i])
//            continue;
//        for(int j = i + 1; j < n_instances; j++){
//            if(in_bundle[j])
//                continue;
//            if(equal(dense_id_host + i * n_feature,
//                     dense_id_host + (i + 1) * n_feature, dense_id_host + j * n_feature)){
//                in_bundle[i] = 1;
//                in_bundle[j] = 1;
//                to_del.push_back(j);
//            }
//        }
//    }
//    std::cout<<"number of instances to be deleted:"<<to_del.size()<<std::endl;
//    if(to_del.size() == 0)
//        return;
//    vector<vector<int>> n_bin_id_after_bundle(n_feature);
//    vector<vector<float_type>> n_new_csc_val(n_feature);
//    vector<vector<int>> n_new_row_ind(n_feature);
//    //vector<vector<int>> new_col_ptr(n_feature);
//
//#pragma omp parallel for
//    for(int i = 0; i < columns.csc_col_ptr.size() - 1; i++){
//        int deled_num = 0;
//        for(int j = col_ptr[i]; j < col_ptr[i + 1]; j++){
//            if(std::find(to_del.begin(), to_del.end(), row_ptr[j]) == to_del.end()){
//                n_new_csc_val[i].push_back(val_ptr[j]);
//                n_new_row_ind[i].push_back(row_ptr[j] - deled_num);
//                n_bin_id_after_bundle[i].push_back(bin_id_data[j]);
//            }
//            else
//                deled_num ++;
//        }
//    }
//    vector<float_type> new_csc_val;
//    vector<int> new_row_ind;
//    vector<int> new_col_ptr;
//    vector<int> bin_id_after_bundle;
//    for(int i = 0; i < n_feature; i++){
//        new_csc_val.insert(new_csc_val.end(), n_new_csc_val[i].begin(), n_new_csc_val[i].end());
//        new_row_ind.insert(new_row_ind.end(), n_new_row_ind[i].begin(), n_new_row_ind[i].end());
//        new_col_ptr.push_back(new_col_ptr.back() + n_new_csc_val[i].size());
//        bin_id_after_bundle.insert(bin_id_after_bundle.end(), n_bin_id_after_bundle[i].begin(), n_bin_id_after_bundle[i].end());
//    }
//    columns.nnz = new_csc_val.size();
//    columns.csc_val.resize(new_csc_val.size());
//    memcpy(columns.csc_val.host_data(), new_csc_val.data(), sizeof(float_type) * new_csc_val.size());
//    columns.csc_row_ind.resize(new_row_ind.size());
//    memcpy(columns.csc_row_ind.host_data(), new_row_ind.data(), sizeof(int) * new_row_ind.size());
//    columns.csc_col_ptr.resize(new_col_ptr.size());
//    memcpy(columns.csc_col_ptr.host_data(), new_col_ptr.data(), sizeof(int) * new_col_ptr.size());
//
//}


void HistUpdater::init_bin_id_csr(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns, int n_instances){
    SparseColumns& columns = *v_columns[0][0];

    //SparseColumns& columns = unsort_columns;
    int n_column = columns.n_column;
    //    std::cout<<"n_column:"<<n_column;
    //    Csc2r bin_id_csr;
    SyncArray<int>& bin_id_val = *bin_id[0];
    //    LOG(INFO)<<"bin id"<<bin_id_val;


    int* bin_id_val_data = bin_id_val.device_data();
    int* bin_id_val_host = bin_id_val.host_data();
    LOG(INFO)<<"bin id val:"<<bin_id_val;
    LOG(INFO)<<"csc col:"<<columns.csc_col_ptr;
    float_type* csc_val_device = columns.csc_val.device_data();
    int* csc_row_device = columns.csc_row_ind.device_data();
    int* csc_col_device = columns.csc_col_ptr.device_data();

    float_type* csc_val_host = columns.csc_val.host_data();
    int* csc_row_host = columns.csc_row_ind.host_data();
    int* csc_col_host = columns.csc_col_ptr.host_data();

    SyncArray<float_type> bin_id_float(bin_id_val.size());
    float_type* bin_id_float_data = bin_id_float.device_data();
    device_loop(bin_id_val.size(), [=]__device__(int bid){
        bin_id_float_data[bid] = bin_id_val_data[bid] * 1.0 + 1.0;
    });
    bin_id_csr.from_csr(bin_id_float.device_data(), csc_row_device, csc_col_device, n_column, n_instances, columns.nnz);
    return;
}


void HistUpdater::similar_ins_bundle(const vector<std::shared_ptr<SparseColumns>> &v_columns,
        InsStat &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
        int* iidold2new, SyncArray<bool>& is_multi){
    using namespace thrust;

    SparseColumns& columns = *v_columns[0];
    //SparseColumns& columns = unsort_columns;
    int n_column = columns.n_column;
//    std::cout<<"n_column:"<<n_column;
    float_type *stats_y = stats.y.host_data();
    SyncArray<int>& bin_id_val = *bin_id[0];
//    LOG(INFO)<<"bin id"<<bin_id_val;


    int* bin_id_val_data = bin_id_val.device_data();
    float_type* csc_val_device = columns.csc_val.device_data();
    int* csc_row_device = columns.csc_row_ind.device_data();
    int* csc_col_device = columns.csc_col_ptr.device_data();

    float_type* csc_val_host = columns.csc_val.host_data();
    int* csc_row_host = columns.csc_row_ind.host_data();
    int* csc_col_host = columns.csc_col_ptr.host_data();

//    float_type* csc_val_device = columns.csc_val.device_data();
//    int* csc_row_device = columns.csc_row_ind.device_data();
//    int* csc_col_device = columns.csc_col_ptr.device_data();
//
//    float_type* csc_val_host = columns.csc_val.host_data();
//    int* csc_row_host = columns.csc_row_ind.host_data();
//    int* csc_col_host = columns.csc_col_ptr.host_data();

//    int* bin_id_val_host = bin_id_val.host_data();
//    std::cout<<"bin id fifth col before transform:"<<std::endl;
//    for(int i = csc_col_host[4]; i < csc_col_host[5]; i++){
//        std::cout<<bin_id_val_host[i]<<"\t";
//    }
//    std::cout<<std::endl;
//    std::cout<<"old bin id (0,0):"<< bin_id_val.host_data()[columns.csc_row_ind.host_data()[0]]<<std::endl;
//    std::cout<<"old bin id (1,0):"<< bin_id_val.host_data()[1]<<std::endl;
//    std::cout<<"test id:"<< ((int *)((float_type *)bin_id_val.host_data()))[0]<<std::endl;
//    std::cout<<"bin id size:"<<bin_id_val.size()<<std::endl;
    SyncArray<float_type> bin_id_float(bin_id_val.size());
    float_type* bin_id_float_data = bin_id_float.device_data();
    device_loop(bin_id_val.size(), [=]__device__(int bid){
        bin_id_float_data[bid] = bin_id_val_data[bid] * 1.0 + 1.0;
    });
//
//    float test = 0;
//    cudaMemcpy(&test, bin_id_float_data, sizeof(float),
//               cudaMemcpyDeviceToHost);
//    int *t = (int *)&test;
    //std::cout<<"test:"<<*t<<std::endl;
//    std::cout<<"n_column:"<<n_column<<std::endl;
//    std::cout<<"n_instances:"<<n_instances<<std::endl;
//    std::cout<<"nnz:"<<columns.nnz<<std::endl;
//    std::cout<<"csc col size:"<<columns.csc_col_ptr.size()<<std::endl;
//    std::cout<<"csc row size:"<<columns.csc_row_ind.size()<<std::endl;
//    LOG(INFO)<<"csc col"<<unsort_columns.csc_col_ptr;
//    LOG(INFO)<<"csc row"<<unsort_columns.csc_row_ind;
//    LOG(INFO)<<"bin id float"<<bin_id_float;
//    std::cout<<"unsort nnz:"<<unsort_columns.nnz;

//    float_type* bin_id_float_host = bin_id_float.host_data();
//    std::cout<<"bin float id fifth col before transform:"<<std::endl;
//    for(int i = csc_col_host[4]; i < csc_col_host[5]; i++){
//        std::cout<<bin_id_float_host[i]<<"\t";
//    }
//    std::cout<<std::endl;

    std::cout<<"number of non-zero values before bundle:"<<unsort_columns.nnz<<std::endl;
    LOG(INFO)<<"bin id float"<<bin_id_float;
    LOG(INFO)<<"csc row"<<unsort_columns.csc_row_ind;
    LOG(INFO)<<"csc col"<<unsort_columns.csc_col_ptr;
    bin_id_csr.from_csr(bin_id_float.device_data(), csc_row_device, csc_col_device, n_column, n_instances, columns.nnz);

    LOG(INFO)<<"bin id csr val:"<<bin_id_csr.csc_val;
    //cudaDeviceSynchronize();
    //std::cout<<"test id:"<< ((int *)((float_type *)bin_id_val.host_data()))[0]<<std::endl;
    //float_type* bin_id_csr_val = bin_id_csr.csc_val.device_data();
    float_type* bin_id_csr_val_host = bin_id_csr.csc_val.host_data();
    int* bin_id_csr_row = bin_id_csr.csc_col_ptr.device_data();
    int* bin_id_csr_col = bin_id_csr.csc_row_ind.device_data();
    int* bin_id_csr_row_host = bin_id_csr.csc_col_ptr.host_data();
    int* bin_id_csr_col_host = bin_id_csr.csc_row_ind.host_data();

//    std::cout<<"bin id fifth col after transform:"<<std::endl;
//    for(int i = 0; i < bin_id_csr.csc_val.size(); i++){
//        if(bin_id_csr_col_host[i] == 4){
//            std::cout<<bin_id_csr_val_host[i]<<"\t";
//        }
//    }
//    std::cout<<std::endl;
//    cudaMemcpy(&test, bin_id_csr.csc_val.device_data(), sizeof(float),
//               cudaMemcpyDeviceToHost);
//    std::cout<<"test t:"<<test<<std::endl;
//    std::cout<<"test:"<<*t<<std::endl;

//    LOG(INFO)<<"bin id csr val"<<bin_id_csr.csc_val;
//    std::cout<<"csr val last row:"<<std::endl;
//    for(int i = bin_id_csr_row_host[n_instances - 1]; i < bin_id_csr_row_host[n_instances]; i++){
//        std::cout<<bin_id_csr_val_host[i]<<"\t";
//    }
//    std::cout<<std::endl;
//    LOG(INFO)<<"bin id csr row"<<bin_id_csr.csc_col_ptr;
//    LOG(INFO)<<"bin id csr col"<<bin_id_csr.csc_row_ind;
//    std::cout<<"val size:"<<bin_id_csr.csc_val.size()<<std::endl;

//    std::cout<<"new bin id (0,0):"<<bin_id_csr.csc_val.host_data()[bin_id_csr.csc_col_ptr.host_data()[0]]<<std::endl;
//    std::cout<<"new bin id (1,0):"<<(bin_id_csr.csc_val.host_data())[bin_id_csr.csc_col_ptr.host_data()[1]]<<std::endl;

    /*
    SyncArray<int> bin_id_first_column_dense(n_instances);
    int* bin_id_first_column_dense_data = bin_id_first_column_dense.device_data();
    int first_column_nnz = columns.csc_col_ptr.host_data()[1];
    device_loop(first_column_nnz, [=]__device__(int bid){
        bin_id_first_column_dense_data[csc_row_device[bid]] = bin_id_val_data[bid];
    });

    SyncArray<int> sorted_row_id(n_instances);
    int * sorted_row_id_data = sorted_row_id.device_data();
    sequence(cuda::par, sorted_row_id.device_data(), sorted_row_id.device_end(), 0);
    //sort by the first column
    stable_sort_by_key(cuda::par, bin_id_first_column_dense_data,
            bin_id_first_column_dense_data + first_column_nnz, sorted_row_id.device_data());


    device_loop(n_instances - 1, [=]__device__(int iid){
        is_del_data[iid] = (equal(cuda::par, bin_id_csr_val + bin_id_csr_row[iid],
                bin_id_csr_val + bin_id_csr_row[iid + 1],
                bin_id_csr_val + bin_id_csr_row[iid + 1])) &&
                        (equal(cuda::par, bin_id_csr_col + bin_id_csr_row[iid],
                                bin_id_csr_col + bin_id_csr_row[iid + 1],
                                bin_id_csr_col + bin_id_csr_row[iid + 1]));
    });
            */


    //bool* is_del_data = is_del.device_data();
    SyncArray<bool> is_del(n_instances);
    bool* is_del_host = is_del.host_data();

    //vector<bool> in_bundle(n_instances);

//    for(int i = 0; i < n_instances - 1; i++){
//        if(in_bundle[i])
//            continue;
//        for(int j = i + 1; j < n_instances; j++){
//            if(in_bundle[j])
//                continue;
//            if(((bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) == (bin_id_csr_row_host[j+1] - bin_id_csr_row_host[j]))
//                && equal(bin_id_csr_val_host + bin_id_csr_row_host[i],
//                         bin_id_csr_val_host + bin_id_csr_row_host[i + 1], bin_id_csr_val_host + bin_id_csr_row_host[j])){
//                in_bundle[i] = 1;
//                in_bundle[j] = 1;
//                is_del_host[j] = 1;
//            }
//        }
//    }


//    SyncArray<int> same_col(n_column);
//    int* same_col_host = same_col.host_data();

    SyncArray<float_type> y_new(n_instances);
    y_new.copy_from(stats_y, n_instances);
    float_type* y_new_host = y_new.host_data();

//    SyncArray<int> col_id_1(n_column);
//    SyncArray<int> col_id_2(n_column);
//    int* col_id_1_host = col_id_1.host_data();
//    int* col_id_2_host = col_id_2.host_data();

    vector<vector<float_type>> new_dense_val(n_instances);
    vector<vector<int>> new_csr_col(n_instances);
    vector<int> nnz_perrow(n_instances);

    bool* is_multi_data = is_multi.host_data();
    int oiid = 0;
    int miss_col = 0;
    int inequal_dis = 0;

    for(int i = 0; i < n_instances; i++){
        if(is_del_host[i])
            continue;
        iidold2new[i] = oiid;

        int* bin_col_i = bin_id_csr_col_host + bin_id_csr_row_host[i];
        float_type* bin_val_i = bin_id_csr_val_host + bin_id_csr_row_host[i];

        new_dense_val[i].resize(n_column, 0.f);

        for(int t = 0; t < bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]; t++){
            new_dense_val[i][bin_col_i[t]] = bin_val_i[t];

        }


        nnz_perrow[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];

        for(int j = i + 1; j < n_instances; j++){
            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) > 2)
                continue;

//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0))
//                continue;

//            int y_diff = fabs(stats_y[i] - stats_y[j]);
            int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
            float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//            bool is_same = 1;
            int same_col_num = 0;

            miss_col = 0;
            inequal_dis = 0;

            for(int m = 0, n = 0; m <  (bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) && n < (bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]);){
                if(bin_col_i[m] < bin_col_j[n]) {
                    m++;
                    miss_col++;

//                    is_same = 0;
                }
                else if(bin_col_i[m] == bin_col_j[n]){
                    if(bin_val_i[m] != bin_val_j[n]){
//                        is_same = 0;
                        inequal_dis += abs((int)(bin_val_i[m] - bin_val_j[n]));
//                        break;
                    }
                    same_col_num ++;
                    m++;
                    n++;
                }
                else {
                    n++;
                    miss_col++;

//                    is_same = 0;
                }
            }
            if((inequal_dis * 5 + miss_col) < 50){
//            if(is_same){
                iidold2new[j] = oiid;
                is_multi_data[oiid] = 1;
                is_del_host[j] = 1;
                y_new_host[i] = y_new_host[i] + y_new_host[j];
                y_new_host[j] = INFINITY;
//                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
//                    new_dense_val[i][bin_col_j[t]] = bin_val_j[t];
//                }
//                nnz_perrow[i] += bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - same_col_num;
                break;
            }
        }
        oiid++;
    }



//    for(int i = 0; i < n_instances; i++){
//        if(is_del_host[i])
//            continue;
//        iidold2new[i] = oiid;
//
//        int* bin_col_i = bin_id_csr_col_host + bin_id_csr_row_host[i];
//        float_type* bin_val_i = bin_id_csr_val_host + bin_id_csr_row_host[i];
//
//        new_dense_val[i].resize(n_column, 0.f);
//
//        for(int t = 0; t < bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]; t++){
//            new_dense_val[i][bin_col_i[t]] = bin_val_i[t];
//
//        }
//
//        nnz_perrow[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];
//
//        int min_dis = 10;
//        for(int j = i + 1; j < n_instances; j++){
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) > 2)
//                continue;
//
//            int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
//            float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//            bool is_same = 1;
//            int same_col_num = 0;
//            for(int m = 0, n = 0; m <  (bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) && n < (bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]);){
//                if(bin_col_i[m] < bin_col_j[n]) {
//                    m++;
//                    miss_col++;
//                }
//                else if(bin_col_i[m] == bin_col_j[n]){
//                    if(bin_val_i[m] != bin_val_j[n]){
//                        is_same = 0;
//                        inequal_dis += abs((int)(bin_val_i[m] - bin_val_j[n]));
//                        break;
//                    }
//                    same_col_num ++;
//                    m++;
//                    n++;
//                }
//                else {
//                    n++;
//                    miss_col++;
//                }
//            }
//            if(inequal_dis < )
//            if((inequal_dis + miss_col * 2) < 10){
//                iidold2new[j] = oiid;
//                is_multi_data[oiid] = 1;
//                is_del_host[j] = 1;
//                y_new_host[i] = y_new_host[i] + y_new_host[j];
//                y_new_host[j] = INFINITY;
////                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
////                    new_dense_val[i][bin_col_j[t]] = bin_val_j[t];
////                }
////                nnz_perrow[i] += bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - same_col_num;
//                break;
//            }
//        }
//        oiid++;
//    }





    int del_size = count(is_del.host_data(), is_del.host_data() + n_instances, 1);
    std::cout << "del size:" << del_size << std::endl;
    //del_size = 0;
    if(del_size != 0) {
        int total_nnz_new = reduce(nnz_perrow.begin(), nnz_perrow.begin() + n_instances);
        std::cout<<"total nonzero new:"<<total_nnz_new<<std::endl;
        vector<float_type> new_csc_val(total_nnz_new, 0.f);
        vector<int> new_csc_row(total_nnz_new);
        vector<int> new_csc_col(n_column + 1);
        new_csc_col[0] = 0;

        vector<float_type> new_dense_val_total;
        int new_row_size = 0;
        for(int i = 0; i < n_instances; i++){
            if(new_dense_val[i].size() != 0){
                new_row_size++;
                new_dense_val_total.insert(new_dense_val_total.end(), new_dense_val[i].begin(), new_dense_val[i].end());
            }
        }
//        std::cout<<"after dense val"<<std::endl;
//
//        std::cout<<"new row size:"<<new_row_size<<std::endl;
        int val_id = 0;
        for(int i = 0; i < n_column; i++){
            for(int j = 0; j < new_row_size; j++){
                if(new_dense_val_total[j * n_column + i] != 0.f) {
                    new_csc_val[val_id] = new_dense_val_total[j * n_column + i] - 1;
                    new_csc_row[val_id++] = j;
                }
            }
            new_csc_col[i + 1] = val_id;
        }
//        std::cout<<"val id:"<<val_id<<std::endl;
//        std::cout<<"after new_csc_val"<<std::endl;
        //bin id
        //columns.csc_val.resize(new_csc_val.size());
        //columns.csc_val.copy_from(new_csc_val.data(), new_csc_val.size());
        columns.csc_val.resize(0);
        columns.csc_bin_id.resize(new_csc_val.size());
        columns.csc_bin_id.copy_from(new_csc_val.data(), new_csc_val.size());
        columns.csc_row_ind.resize(new_csc_row.size());
        columns.csc_row_ind.copy_from(new_csc_row.data(), new_csc_row.size());
        columns.csc_col_ptr.resize(new_csc_col.size());
        columns.csc_col_ptr.copy_from(new_csc_col.data(), new_csc_col.size());
        columns.nnz = total_nnz_new;

//        LOG(INFO)<<"bin_id before sort"<<columns.csc_bin_id;


//        auto bin_id_val_before_sort = columns.csc_bin_id.host_data();
//        auto col_host_before_sort = columns.csc_col_ptr.host_data();
//        std::cout<<"bin id fifth col before sort:"<<std::endl;
//        for(int i = col_host_before_sort[4]; i < col_host_before_sort[5]; i++){
//            std::cout<<bin_id_val_before_sort[i]<<"\t";
//        }
//        std::cout<<std::endl;
//
//        auto row_ind_before_sort = columns.csc_row_ind.host_data();
//        std::cout<<"csc row fifth col before sort:"<<std::endl;
//        for(int i = col_host_before_sort[4]; i < col_host_before_sort[5]; i++){
//            std::cout<<row_ind_before_sort[i]<<"\t";
//        }
//        std::cout<<std::endl;

        //sort bin ids after instance bundle
        //DO_ON_MULTI_DEVICES(1, [&](int device_id) {
            cub_seg_sort_by_key(columns.csc_bin_id, columns.csc_row_ind, columns.csc_col_ptr, true);
        //});
        //cudaDeviceSynchronize();
//        LOG(INFO)<<"bin_id after sort"<<columns.csc_bin_id;
//        LOG(INFO)<<"csc row after sort"<<columns.csc_row_ind;
//        LOG(INFO)<<"csc col after sort"<<columns.csc_col_ptr;
//
//        std::cout<<"bin id fifth col after sort:"<<std::endl;
//        for(int i = col_host_before_sort[4]; i < col_host_before_sort[5]; i++){
//            std::cout<<bin_id_val_before_sort[i]<<"\t";
//        }
//        std::cout<<std::endl;
//
//        std::cout<<"csc row fifth col after sort:"<<std::endl;
//        for(int i = col_host_before_sort[4]; i < col_host_before_sort[5]; i++){
//            std::cout<<row_ind_before_sort[i]<<"\t";
//        }
//        std::cout<<std::endl;
//        cusparseHandle_t handle;
//        cusparseCreate(&handle);
//        cusparseMatDescr_t descr;
//        cusparseCreateMatDescr(&descr);
//        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
//        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//
//        //row-major, convert to csr
//        cusparseSdense2csc(handle, n_column, n_instances - del_size, n_column, );
//        cudaDeviceSynchronize();
//        cusparseDestroy(handle);



//        for(int i = n_instances - 1; i >= 0; i--){
//            if(is_del[i]){
//                std::erase(dataSet.instances_.begin() + i);
//
//            }
//        }


        float_type *y_new_end = remove(y_new_host, y_new_host + n_instances, INFINITY);
        CHECK(del_size == (n_instances - (y_new_end - y_new_host)));
//        std::cout<<"old n_instances: "<<n_instances<<std::endl;
//        std::cout << "del size:" << del_size << std::endl;
        n_instances = n_instances - del_size;
//        std::cout<<"new n_instances:"<<n_instances<<std::endl;
        stats.resize(n_instances);

        stats.y.copy_from(y_new_host, y_new_end - y_new_host);
        stats.updateGH(is_multi);
//        LOG(INFO)<<"new stat y"<<stats.y;
        std::cout<<"new stats n_instance: "<<stats.n_instances<<std::endl;






    }
}

void HistUpdater::similar_ins_bundle(const vector<std::shared_ptr<SparseColumns>> &v_columns,
                                     const vector<std::shared_ptr<SparseColumns>> &v_columns2,
                                     InsStat &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
                                     int* iidold2new, SyncArray<bool>& is_multi){
    using namespace thrust;

    SparseColumns& columns = *v_columns[0];
    SparseColumns& columns2 = *v_columns2[0];
    //SparseColumns& columns = unsort_columns;
    int n_column = columns.n_column;
//    std::cout<<"n_column:"<<n_column;
    float_type *stats_y = stats.y.host_data();
    Csc2r bin_id_csr;
    SyncArray<int>& bin_id_val = *bin_id[0];
//    LOG(INFO)<<"bin id"<<bin_id_val;


    int* bin_id_val_data = bin_id_val.device_data();
    float_type* csc_val_device = columns.csc_val.device_data();
    int* csc_row_device = columns.csc_row_ind.device_data();
    int* csc_col_device = columns.csc_col_ptr.device_data();

    float_type* csc_val_host = columns.csc_val.host_data();
    int* csc_row_host = columns.csc_row_ind.host_data();
    int* csc_col_host = columns.csc_col_ptr.host_data();

    SyncArray<float_type> bin_id_float(bin_id_val.size());
    float_type* bin_id_float_data = bin_id_float.device_data();
    device_loop(bin_id_val.size(), [=]__device__(int bid){
        bin_id_float_data[bid] = bin_id_val_data[bid] * 1.0 + 1.0;
    });


    std::cout<<"number of non-zero values before bundle:"<<unsort_columns.nnz<<std::endl;
    LOG(INFO)<<"bin id float"<<bin_id_float;
    LOG(INFO)<<"csc row"<<unsort_columns.csc_row_ind;
    LOG(INFO)<<"csc col"<<unsort_columns.csc_col_ptr;
    bin_id_csr.from_csr(bin_id_float.device_data(), csc_row_device, csc_col_device, n_column, n_instances, columns.nnz);

    LOG(INFO)<<"bin id csr val:"<<bin_id_csr.csc_val;
    //cudaDeviceSynchronize();
    //std::cout<<"test id:"<< ((int *)((float_type *)bin_id_val.host_data()))[0]<<std::endl;
    //float_type* bin_id_csr_val = bin_id_csr.csc_val.device_data();
    float_type* bin_id_csr_val_host = bin_id_csr.csc_val.host_data();
    int* bin_id_csr_row = bin_id_csr.csc_col_ptr.device_data();
    int* bin_id_csr_col = bin_id_csr.csc_row_ind.device_data();
    int* bin_id_csr_row_host = bin_id_csr.csc_col_ptr.host_data();
    int* bin_id_csr_col_host = bin_id_csr.csc_row_ind.host_data();




    //bool* is_del_data = is_del.device_data();
    SyncArray<bool> is_del(n_instances);
    bool* is_del_host = is_del.host_data();



//    SyncArray<int> same_col(n_column);
//    int* same_col_host = same_col.host_data();

    SyncArray<float_type> y_new(n_instances);
    y_new.copy_from(stats_y, n_instances);
    float_type* y_new_host = y_new.host_data();

//    SyncArray<int> col_id_1(n_column);
//    SyncArray<int> col_id_2(n_column);
//    int* col_id_1_host = col_id_1.host_data();
//    int* col_id_2_host = col_id_2.host_data();

    vector<vector<float_type>> new_dense_val(n_instances);
    vector<vector<int>> new_csr_col(n_instances);
    vector<int> nnz_perrow(n_instances);
    vector<int> nnz_perrow2(n_instances);

    bool* is_multi_data = is_multi.host_data();
    int oiid = 0;
    int miss_col = 0;
    int inequal_dis = 0;


//    vector<int> instypeid(n_instances, 0); //0 for no similar instance, 1 for front similar, 2 for back similar

    vector<int> similid(n_instances, 0);

    for(int i = 0; i < n_instances; i++){
        if(is_del_host[i])
            continue;
        iidold2new[i] = oiid;

        int* bin_col_i = bin_id_csr_col_host + bin_id_csr_row_host[i];
        float_type* bin_val_i = bin_id_csr_val_host + bin_id_csr_row_host[i];

        new_dense_val[i].resize(n_column, 0.f);

        for(int t = 0; t < bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]; t++){
            new_dense_val[i][bin_col_i[t]] = bin_val_i[t];

        }


        nnz_perrow[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];
        nnz_perrow2[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];


        for(int j = i + 1; j < n_instances; j++){
            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) > 2)
                continue;
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0))
//                continue;
            int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
            float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//            bool is_same = 1;
            int same_col_num = 0;

            miss_col = 0;
            inequal_dis = 0;

            for(int m = 0, n = 0; m <  (bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) && n < (bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]);){
                if(bin_col_i[m] < bin_col_j[n]) {
                    m++;
                    miss_col++;

//                    is_same = 0;
                }
                else if(bin_col_i[m] == bin_col_j[n]){
                    if(bin_val_i[m] != bin_val_j[n]){
//                        is_same = 0;
                        inequal_dis += abs((int)(bin_val_i[m] - bin_val_j[n]));
//                        break;
                    }
                    same_col_num ++;
                    m++;
                    n++;
                }
                else {
                    n++;
                    miss_col++;

//                    is_same = 0;
                }
            }
            if((inequal_dis * 5 + miss_col) < 50){
//            if(is_same){
                iidold2new[j] = oiid;
                is_multi_data[oiid] = 1;
                is_del_host[j] = 1;
                y_new_host[i] = y_new_host[i] + y_new_host[j];
                y_new_host[j] = INFINITY;


                new_dense_val[j].resize(n_column, 0.f);

                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
                    new_dense_val[j][bin_col_j[t]] = bin_val_j[t];

                }
//                instypeid[i] = 1;
//                instypeid[j] = 2;

                similid[i] = j;
                similid[j] = -1;

                nnz_perrow2[i] = bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j];
//                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
//                    new_dense_val[i][bin_col_j[t]] = bin_val_j[t];
//                }
//                nnz_perrow[i] += bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - same_col_num;

                break;
            }
        }
        oiid++;
    }


    int del_size = count(is_del.host_data(), is_del.host_data() + n_instances, 1);
    std::cout << "del size:" << del_size << std::endl;
    //del_size = 0;
    if(del_size != 0) {
        int total_nnz_new = reduce(nnz_perrow.begin(), nnz_perrow.begin() + n_instances);
        int total_nnz_new2 = reduce(nnz_perrow2.begin(), nnz_perrow2.begin() + n_instances);

        std::cout<<"total nonzero new:"<<total_nnz_new<<std::endl;
        std::cout<<"total nonzero new2:"<<total_nnz_new2<<std::endl;
        vector<float_type> new_csc_val(total_nnz_new, 0.f);
        vector<int> new_csc_row(total_nnz_new);
        vector<int> new_csc_col(n_column + 1);
        new_csc_col[0] = 0;

        vector<float_type> new_csc_val2(total_nnz_new2, 0.f);
        vector<int> new_csc_row2(total_nnz_new2);
        vector<int> new_csc_col2(n_column + 1);
        new_csc_col2[0] = 0;

        vector<float_type> new_dense_val_total;
        vector<float_type> new_dense_val_total2;

        int new_row_size = 0;
        for(int i = 0; i < n_instances; i++){
            if(similid[i] == 0){
                new_row_size++;
                new_dense_val_total.insert(new_dense_val_total.end(), new_dense_val[i].begin(), new_dense_val[i].end());
                new_dense_val_total2.insert(new_dense_val_total2.end(), new_dense_val[i].begin(), new_dense_val[i].end());
            }
            else if(similid[i] != -1){
                new_row_size++;
                new_dense_val_total.insert(new_dense_val_total.end(), new_dense_val[i].begin(), new_dense_val[i].end());
                new_dense_val_total2.insert(new_dense_val_total2.end(), new_dense_val[similid[i]].begin(), new_dense_val[similid[i]].end());
            }
//            if(similid[i] != -1){
//                new_row_size++;
//                new_dense_val_total.insert(new_dense_val_total.end(), new_dense_val[i].begin(), new_dense_val[i].end());
//            }

//            if(new_dense_val[i].size() != 0){
//                new_row_size++;
//                new_dense_val_total.insert(new_dense_val_total.end(), new_dense_val[i].begin(), new_dense_val[i].end());
//            }
        }

        int val_id = 0;
        int val_id2 = 0;
        for(int i = 0; i < n_column; i++){
            for(int j = 0; j < new_row_size; j++){
                if(new_dense_val_total[j * n_column + i] != 0.f) {
                    new_csc_val[val_id] = new_dense_val_total[j * n_column + i] - 1;
                    new_csc_row[val_id++] = j;
                }
                if(new_dense_val_total2[j * n_column + i] != 0.f){
                    new_csc_val2[val_id2] = new_dense_val_total2[j * n_column + i] - 1;
                    new_csc_row2[val_id2++] = j;
                }
            }
            new_csc_col[i + 1] = val_id;
            new_csc_col2[i + 1] = val_id2;
        }

        columns.csc_val.resize(0);
        columns.csc_bin_id.resize(new_csc_val.size());
        columns.csc_bin_id.copy_from(new_csc_val.data(), new_csc_val.size());
        columns.csc_row_ind.resize(new_csc_row.size());
        columns.csc_row_ind.copy_from(new_csc_row.data(), new_csc_row.size());
        columns.csc_col_ptr.resize(new_csc_col.size());
        columns.csc_col_ptr.copy_from(new_csc_col.data(), new_csc_col.size());
        columns.nnz = total_nnz_new;

        columns2.csc_val.resize(0);
        columns2.csc_bin_id.resize(new_csc_val2.size());
        columns2.csc_bin_id.copy_from(new_csc_val2.data(), new_csc_val2.size());
        columns2.csc_row_ind.resize(new_csc_row2.size());
        columns2.csc_row_ind.copy_from(new_csc_row2.data(), new_csc_row2.size());
        columns2.csc_col_ptr.resize(new_csc_col2.size());
        columns2.csc_col_ptr.copy_from(new_csc_col2.data(), new_csc_col2.size());
        columns2.nnz = total_nnz_new2;
        columns2.n_column = n_column;


        cub_seg_sort_by_key(columns.csc_bin_id, columns.csc_row_ind, columns.csc_col_ptr, true);


        // to be correct
        cub_seg_sort_by_key(columns2.csc_bin_id, columns2.csc_row_ind, columns2.csc_col_ptr, true);


        float_type *y_new_end = remove(y_new_host, y_new_host + n_instances, INFINITY);
        CHECK(del_size == (n_instances - (y_new_end - y_new_host)));
//        std::cout<<"old n_instances: "<<n_instances<<std::endl;
//        std::cout << "del size:" << del_size << std::endl;
        n_instances = n_instances - del_size;
//        std::cout<<"new n_instances:"<<n_instances<<std::endl;
        stats.resize(n_instances);

        stats.y.copy_from(y_new_host, y_new_end - y_new_host);
        stats.updateGH(is_multi);
//        LOG(INFO)<<"new stat y"<<stats.y;
        std::cout<<"new stats n_instance: "<<stats.n_instances<<std::endl;

    }
}

void HistUpdater::similar_ins_bundle_multi(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns,
        int numP, InsStat &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
        int* iidold2new, SyncArray<bool>& is_multi, bool is_random){

    using namespace thrust;

    SparseColumns& columns = *v_columns[0][0];

    //SparseColumns& columns = unsort_columns;
    int n_column = columns.n_column;
//    std::cout<<"n_column:"<<n_column;
    float_type *stats_y = stats.y.host_data();
    Csc2r bin_id_csr;
    SyncArray<int>& bin_id_val = *bin_id[0];
//    LOG(INFO)<<"bin id"<<bin_id_val;


    int* bin_id_val_data = bin_id_val.device_data();
    int* bin_id_val_host = bin_id_val.host_data();
    LOG(INFO)<<"bin id val:"<<bin_id_val;
    LOG(INFO)<<"csc col:"<<columns.csc_col_ptr;
    float_type* csc_val_device = columns.csc_val.device_data();
    int* csc_row_device = columns.csc_row_ind.device_data();
    int* csc_col_device = columns.csc_col_ptr.device_data();

    float_type* csc_val_host = columns.csc_val.host_data();
    int* csc_row_host = columns.csc_row_ind.host_data();
    int* csc_col_host = columns.csc_col_ptr.host_data();

    SyncArray<float_type> bin_id_float(bin_id_val.size());
    float_type* bin_id_float_data = bin_id_float.device_data();
    device_loop(bin_id_val.size(), [=]__device__(int bid){
        bin_id_float_data[bid] = bin_id_val_data[bid] * 1.0 + 1.0;
    });


    std::cout<<"number of non-zero values before bundle:"<<unsort_columns.nnz<<std::endl;
    LOG(INFO)<<"bin id float"<<bin_id_float;
    LOG(INFO)<<"csc row"<<unsort_columns.csc_row_ind;
    LOG(INFO)<<"csc col"<<unsort_columns.csc_col_ptr;
    bin_id_csr.from_csr(bin_id_float.device_data(), csc_row_device, csc_col_device, n_column, n_instances, columns.nnz);

    LOG(INFO)<<"bin id csr val:"<<bin_id_csr.csc_val;

    float_type* bin_id_csr_val_host = bin_id_csr.csc_val.host_data();
    int* bin_id_csr_row = bin_id_csr.csc_col_ptr.device_data();
    int* bin_id_csr_col = bin_id_csr.csc_row_ind.device_data();
    int* bin_id_csr_row_host = bin_id_csr.csc_col_ptr.host_data();
    int* bin_id_csr_col_host = bin_id_csr.csc_row_ind.host_data();




    //bool* is_del_data = is_del.device_data();
    SyncArray<bool> is_del(n_instances);
    bool* is_del_host = is_del.host_data();



//    SyncArray<int> same_col(n_column);
//    int* same_col_host = same_col.host_data();

    SyncArray<float_type> y_new(n_instances);
    y_new.copy_from(stats_y, n_instances);
    float_type* y_new_host = y_new.host_data();


    vector<vector<float_type>> new_dense_val(n_instances);
    vector<vector<int>> new_csr_col(n_instances);

    vector<int> nnz_perrow(n_instances, 0);
    vector<int> nnz_perrow_diff(numP, 0);
    vector<int> nnz_perrow_diff_current(numP, 0);


    bool* is_multi_data = is_multi.host_data();
    int oiid = 0;
    int miss_col = 0;
    int inequal_dis = 0;
    float dis_percen = 0;

    srand (time(NULL));
    vector<int> instypeid(n_instances, 0); //0 for no similar instance, 1 for front similar, 2 for back similar


    vector<vector<int>> similid(n_instances);

    for(int i = 0; i < n_instances; i++){
        if(is_del_host[i])
            continue;
        iidold2new[i] = oiid;

        int* bin_col_i = bin_id_csr_col_host + bin_id_csr_row_host[i];
        float_type* bin_val_i = bin_id_csr_val_host + bin_id_csr_row_host[i];

        new_dense_val[i].resize(n_column, 0.f);

        for(int t = 0; t < bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]; t++){
            new_dense_val[i][bin_col_i[t]] = bin_val_i[t];

        }


        nnz_perrow[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];
        //nnz_perrow2[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];

        similid[i].push_back(i);

        int nparts = 1;
        bool is_out = 0;
        for(int j = i + 1; j < n_instances; j++){
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) != 0)
//                continue;
            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) > 2)
                continue;
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0))
//                continue;
            float_type y_diff = fabs(stats_y[i] - stats_y[j]);
            int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
            float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//            bool is_same = 1;
            int same_col_num = 0;

            miss_col = 0;
            inequal_dis = 0;

            is_out = 0;
            dis_percen = 0;
            for(int m = 0, n = 0; m <  (bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) && n < (bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]);){
                if(bin_col_i[m] < bin_col_j[n]) {
                    m++;
                    miss_col++;
//                    is_same = 0;
                }
                else if(bin_col_i[m] == bin_col_j[n]){
                    if(bin_val_i[m] != bin_val_j[n]){
//                        is_same = 0;
                        inequal_dis += abs((int)(bin_val_i[m] - bin_val_j[n]));
                        float percen = 1.0 * abs((int)(bin_val_i[m] - bin_val_j[n])) / (bin_id_val_host[csc_col_host[bin_col_i[m] + 1] - 1] + 1);
                        CHECK(percen <= 1.f)<<percen;
                        if(((bin_id_val_host[csc_col_host[bin_col_i[m] + 1] - 1] > 10) && (percen > 0.1))){
                            is_out = 1;
                            break;
                        }
                        dis_percen += percen;
//                        break;
                    }
                    same_col_num ++;
                    m++;
                    n++;
                }
                else {
                    n++;
                    miss_col++;

//                    is_same = 0;
                }
            }
//            if((inequal_dis*5 + miss_col) < 100){
            if(((!is_random) && (is_out == 0) && (dis_percen < 1) && (miss_col <= (n_column / 10))) ||
                (is_random && ((rand() % 100) == 1))){
//            if((rand() % 100) == 1){
//            if(j == (i + n_instances / 2)){

//            if(is_same){

                nparts++;
                iidold2new[j] = oiid;
                is_multi_data[oiid] = 1;
                is_del_host[j] = 1;
                y_new_host[i] = y_new_host[i] + y_new_host[j];
                y_new_host[j] = INFINITY;


                new_dense_val[j].resize(n_column, 0.f);

                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
                    new_dense_val[j][bin_col_j[t]] = bin_val_j[t];

                }

                similid[i].push_back(j);

                nnz_perrow_diff_current[nparts - 1] = bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];
                nnz_perrow_diff[nparts - 1] += bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];

                if(nparts == numP)
                    break;
            }
        }
        if(nparts < numP){
            for(int pid = nparts; pid < numP; pid++){
                similid[i].push_back(similid[i][pid % nparts]);
                nnz_perrow_diff[pid] += nnz_perrow_diff_current[pid % nparts];
            }
        }
        oiid++;
    }


    int del_size = count(is_del.host_data(), is_del.host_data() + n_instances, 1);
    std::cout << "del size:" << del_size << std::endl;
    //del_size = 0;
    if(del_size != 0) {
        int total_nnz_new = reduce(nnz_perrow.begin(), nnz_perrow.begin() + n_instances);

        vector<vector<float_type>> new_csc_val(numP);
        vector<vector<int>> new_csc_row(numP);
        vector<vector<int>> new_csc_col(numP);

#pragma omp parallel for
        for(int i = 0; i < numP; i++){
            new_csc_val[i].resize(total_nnz_new + nnz_perrow_diff[i], 0.f);
            new_csc_row[i].resize(total_nnz_new + nnz_perrow_diff[i]);
            new_csc_col[i].resize(n_column + 1);
            new_csc_col[i][0] = 0;
        }

        vector<vector<float_type>> new_dense_val_total(numP);
//        std::cout<<"vector max size:"<<new_dense_val_total.max_size()<<std::endl;
        int new_row_size = 0;
        for(int i = 0; i < n_instances; i++){
            if(similid[i].size() != 0){
                new_row_size++;
                CHECK(similid[i].size() == numP)<<similid[i].size();
                for(int pid = 0; pid < similid[i].size(); pid++){
                    new_dense_val_total[pid].insert(new_dense_val_total[pid].end(), new_dense_val[similid[i][pid]].begin(), new_dense_val[similid[i][pid]].end());
                }

            }
        }

        vector<int> val_id(numP, 0);
        for(int i = 0; i < n_column; i++){
            for(int j = 0; j < new_row_size; j++){
                for(int pid = 0; pid < numP; pid++){
                    if(new_dense_val_total[pid][j * n_column + i] != 0.f){
                        new_csc_val[pid][val_id[pid]] = new_dense_val_total[pid][j * n_column + i] - 1;
                        new_csc_row[pid][val_id[pid]++] = j;
                    }
                }

            }
            for(int pid = 0; pid < numP; pid++){
                new_csc_col[pid][i + 1] = val_id[pid];
            }
        }


// to do parallel
        for(int pid = 0; pid < numP; pid++){
            (*v_columns[pid][0]).csc_bin_id.resize(new_csc_val[pid].size());
            (*v_columns[pid][0]).csc_bin_id.copy_from(new_csc_val[pid].data(), new_csc_val[pid].size());
            (*v_columns[pid][0]).csc_row_ind.resize(new_csc_row[pid].size());
            (*v_columns[pid][0]).csc_row_ind.copy_from(new_csc_row[pid].data(), new_csc_row[pid].size());
            (*v_columns[pid][0]).csc_col_ptr.resize(new_csc_col[pid].size());
            (*v_columns[pid][0]).csc_col_ptr.copy_from(new_csc_col[pid].data(), new_csc_col[pid].size());
            (*v_columns[pid][0]).nnz = total_nnz_new + nnz_perrow_diff[pid];
            (*v_columns[pid][0]).n_column = n_column;
            cub_seg_sort_by_key((*v_columns[pid][0]).csc_bin_id, (*v_columns[pid][0]).csc_row_ind, (*v_columns[pid][0]).csc_col_ptr, true);
            cudaDeviceSynchronize();
        }

        float_type *y_new_end = remove(y_new_host, y_new_host + n_instances, INFINITY);
        CHECK(del_size == (n_instances - (y_new_end - y_new_host)));
//        std::cout<<"old n_instances: "<<n_instances<<std::endl;
//        std::cout << "del size:" << del_size << std::endl;
        n_instances = n_instances - del_size;
//        std::cout<<"new n_instances:"<<n_instances<<std::endl;
        stats.resize(n_instances);

        stats.y.copy_from(y_new_host, y_new_end - y_new_host);
//        stats.updateGH();
        stats.updateGH(is_multi, numP);
//        stats.updateGH(is_multi);
//        LOG(INFO)<<"new stat y"<<stats.y;
        std::cout<<"new stats n_instance: "<<stats.n_instances<<std::endl;
//        LOG(INFO)<<is_multi;

    }
};

void HistUpdater::similar_ins_bundle_closest(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns,
                                        int numP,
                                        InsStat &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
                                        int* iidold2new, SyncArray<bool>& is_multi){
        using namespace thrust;

        SparseColumns& columns = *v_columns[0][0];

        //SparseColumns& columns = unsort_columns;
        int n_column = columns.n_column;
//    std::cout<<"n_column:"<<n_column;
        float_type *stats_y = stats.y.host_data();
        Csc2r bin_id_csr;
        SyncArray<int>& bin_id_val = *bin_id[0];
//    LOG(INFO)<<"bin id"<<bin_id_val;


        int* bin_id_val_data = bin_id_val.device_data();
        int* bin_id_val_host = bin_id_val.host_data();
        float_type* csc_val_device = columns.csc_val.device_data();
        int* csc_row_device = columns.csc_row_ind.device_data();
        int* csc_col_device = columns.csc_col_ptr.device_data();

        float_type* csc_val_host = columns.csc_val.host_data();
        int* csc_row_host = columns.csc_row_ind.host_data();
        int* csc_col_host = columns.csc_col_ptr.host_data();

        SyncArray<float_type> bin_id_float(bin_id_val.size());
        float_type* bin_id_float_data = bin_id_float.device_data();
        device_loop(bin_id_val.size(), [=]__device__(int bid){
            bin_id_float_data[bid] = bin_id_val_data[bid] * 1.0 + 1.0;
        });


        std::cout<<"number of non-zero values before bundle:"<<unsort_columns.nnz<<std::endl;
        LOG(INFO)<<"bin id float"<<bin_id_float;
        LOG(INFO)<<"csc row"<<unsort_columns.csc_row_ind;
        LOG(INFO)<<"csc col"<<unsort_columns.csc_col_ptr;
        bin_id_csr.from_csr(bin_id_float.device_data(), csc_row_device, csc_col_device, n_column, n_instances, columns.nnz);

        LOG(INFO)<<"bin id csr val:"<<bin_id_csr.csc_val;

        float_type* bin_id_csr_val_host = bin_id_csr.csc_val.host_data();
        int* bin_id_csr_row = bin_id_csr.csc_col_ptr.device_data();
        int* bin_id_csr_col = bin_id_csr.csc_row_ind.device_data();
        int* bin_id_csr_row_host = bin_id_csr.csc_col_ptr.host_data();
        int* bin_id_csr_col_host = bin_id_csr.csc_row_ind.host_data();




        //bool* is_del_data = is_del.device_data();
        SyncArray<bool> is_del(n_instances);
        bool* is_del_host = is_del.host_data();



//    SyncArray<int> same_col(n_column);
//    int* same_col_host = same_col.host_data();

        SyncArray<float_type> y_new(n_instances);
        y_new.copy_from(stats_y, n_instances);
        float_type* y_new_host = y_new.host_data();


        vector<vector<float_type>> new_dense_val(n_instances);
        vector<vector<int>> new_csr_col(n_instances);

        vector<int> nnz_perrow(n_instances, 0);
        vector<int> nnz_perrow_diff(numP, 0);
        vector<int> nnz_perrow_diff_current(numP, 0);


        bool* is_multi_data = is_multi.host_data();
        int oiid = 0;
        int miss_col = 0;
        int inequal_dis = 0;

        srand (time(NULL));
        vector<int> instypeid(n_instances, 0); //0 for no similar instance, 1 for front similar, 2 for back similar


        vector<vector<int>> similid(n_instances);

        for(int i = 0; i < n_instances; i++){
            if(is_del_host[i])
                continue;
            iidold2new[i] = oiid;

            int* bin_col_i = bin_id_csr_col_host + bin_id_csr_row_host[i];
            float_type* bin_val_i = bin_id_csr_val_host + bin_id_csr_row_host[i];

            new_dense_val[i].resize(n_column, 0.f);

            for(int t = 0; t < bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]; t++){
                new_dense_val[i][bin_col_i[t]] = bin_val_i[t];

            }


            nnz_perrow[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];
            //nnz_perrow2[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];

            similid[i].push_back(i);

            int nparts = 1;
            int small_dis = 100;
            int small_ind;

            float small_new_dis = 10;
//            int small_new_ind;

//            float small_p_new_dis[numP-1];
//            int small_new_ind[numP-1];
            for(int j = i + 1; j < n_instances; j++){
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) != 0)
//                continue;
                if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) > 2)
                    continue;
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0))
//                continue;
                float_type y_diff = fabs(stats_y[i] - stats_y[j]);
                int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
                float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//                bool is_same = 1;
                int same_col_num = 0;

                miss_col = 0;
                inequal_dis = 0;

                float new_dis = 0;



                for(int m = 0, n = 0; m <  (bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) && n < (bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]);){
                    if(bin_col_i[m] < bin_col_j[n]) {
                        m++;
                        miss_col++;

//                    is_same = 0;
                    }
                    else if(bin_col_i[m] == bin_col_j[n]){
                        if(bin_val_i[m] != bin_val_j[n]){
//                            is_same = 0;
                            inequal_dis += abs((int)(bin_val_i[m] - bin_val_j[n]));
                            new_dis += 1.0 * abs((int)(bin_val_i[m] - bin_val_j[n])) / (bin_id_val_host[csc_col_host[bin_col_i[m] + 1] - 1] + 1);
//                        break;
                        }
                        same_col_num ++;
                        m++;
                        n++;
                    }
                    else {
                        n++;
                        miss_col++;

//                    is_same = 0;
                    }

                }

                if((inequal_dis*5+ miss_col) < small_dis){
                    small_dis = inequal_dis*5 + miss_col;
                    small_ind = j;
                }

                if((new_dis + miss_col * 0.1) < small_new_dis){
                    small_new_dis = new_dis + miss_col * 0.1;
//                    small_new_ind = j;
                }


            }

            if(small_dis < 100){
              int j = small_ind;
//            if(small_new_dis < 10){
//                int j = small_new_ind;
                int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
                float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//            if((rand() % 100) == 1){
//            if(j == (i + n_instances / 2)){

//            if(is_same){

                nparts++;
                iidold2new[j] = oiid;
                is_multi_data[oiid] = 1;
                is_del_host[j] = 1;
                y_new_host[i] = y_new_host[i] + y_new_host[j];
                y_new_host[j] = INFINITY;


                new_dense_val[j].resize(n_column, 0.f);

                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
                    new_dense_val[j][bin_col_j[t]] = bin_val_j[t];

                }

                similid[i].push_back(j);

                nnz_perrow_diff_current[nparts - 1] = bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];
                nnz_perrow_diff[nparts - 1] += bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];

//                if(nparts == numP)
//                    break;
            }

            if(nparts < numP){
                for(int pid = nparts; pid < numP; pid++){
                    similid[i].push_back(similid[i][pid % nparts]);
                    nnz_perrow_diff[pid] += nnz_perrow_diff_current[pid % nparts];
                }
            }
            oiid++;
        }


        int del_size = count(is_del.host_data(), is_del.host_data() + n_instances, 1);
        std::cout << "del size:" << del_size << std::endl;
        //del_size = 0;
        if(del_size != 0) {
            int total_nnz_new = reduce(nnz_perrow.begin(), nnz_perrow.begin() + n_instances);

            vector<vector<float_type>> new_csc_val(numP);
            vector<vector<int>> new_csc_row(numP);
            vector<vector<int>> new_csc_col(numP);

#pragma omp parallel for
            for(int i = 0; i < numP; i++){
                new_csc_val[i].resize(total_nnz_new + nnz_perrow_diff[i], 0.f);
                new_csc_row[i].resize(total_nnz_new + nnz_perrow_diff[i]);
                new_csc_col[i].resize(n_column + 1);
                new_csc_col[i][0] = 0;
            }

            vector<vector<float_type>> new_dense_val_total(numP);
            int new_row_size = 0;
            for(int i = 0; i < n_instances; i++){
                if(similid[i].size() != 0){
                    new_row_size++;
                    CHECK(similid[i].size() == numP)<<similid[i].size();
                    for(int pid = 0; pid < similid[i].size(); pid++){
                        new_dense_val_total[pid].insert(new_dense_val_total[pid].end(), new_dense_val[similid[i][pid]].begin(), new_dense_val[similid[i][pid]].end());
                    }

                }
            }

            vector<int> val_id(numP, 0);
            for(int i = 0; i < n_column; i++){
                for(int j = 0; j < new_row_size; j++){
                    for(int pid = 0; pid < numP; pid++){
                        if(new_dense_val_total[pid][j * n_column + i] != 0.f){
                            new_csc_val[pid][val_id[pid]] = new_dense_val_total[pid][j * n_column + i] - 1;
                            new_csc_row[pid][val_id[pid]++] = j;
                        }
                    }

                }
                for(int pid = 0; pid < numP; pid++){
                    new_csc_col[pid][i + 1] = val_id[pid];
                }
            }


// to do parallel
            for(int pid = 0; pid < numP; pid++){
                (*v_columns[pid][0]).csc_bin_id.resize(new_csc_val[pid].size());
                (*v_columns[pid][0]).csc_bin_id.copy_from(new_csc_val[pid].data(), new_csc_val[pid].size());
                (*v_columns[pid][0]).csc_row_ind.resize(new_csc_row[pid].size());
                (*v_columns[pid][0]).csc_row_ind.copy_from(new_csc_row[pid].data(), new_csc_row[pid].size());
                (*v_columns[pid][0]).csc_col_ptr.resize(new_csc_col[pid].size());
                (*v_columns[pid][0]).csc_col_ptr.copy_from(new_csc_col[pid].data(), new_csc_col[pid].size());
                (*v_columns[pid][0]).nnz = total_nnz_new + nnz_perrow_diff[pid];
                (*v_columns[pid][0]).n_column = n_column;
                cub_seg_sort_by_key((*v_columns[pid][0]).csc_bin_id, (*v_columns[pid][0]).csc_row_ind, (*v_columns[pid][0]).csc_col_ptr, true);
                cudaDeviceSynchronize();
            }

            float_type *y_new_end = remove(y_new_host, y_new_host + n_instances, INFINITY);
            CHECK(del_size == (n_instances - (y_new_end - y_new_host)));
//        std::cout<<"old n_instances: "<<n_instances<<std::endl;
//        std::cout << "del size:" << del_size << std::endl;
            n_instances = n_instances - del_size;
//        std::cout<<"new n_instances:"<<n_instances<<std::endl;
            stats.resize(n_instances);

            stats.y.copy_from(y_new_host, y_new_end - y_new_host);
            stats.updateGH(is_multi, numP);
//        stats.updateGH(is_multi);
//        LOG(INFO)<<"new stat y"<<stats.y;
            std::cout<<"new stats n_instance: "<<stats.n_instances<<std::endl;

        }
};

void HistUpdater::similar_ins_bundle_independent(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns,
        int numP,
        vector<InsStat> &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
        int* iidold2new, SyncArray<bool>& is_multi, bool is_random, bool weighted_gh){
    using namespace thrust;

    SparseColumns& columns = *v_columns[0][0];

    //SparseColumns& columns = unsort_columns;
    int n_column = columns.n_column;
//    std::cout<<"n_column:"<<n_column;
    float_type *stats_y = stats[0].y.host_data();
    Csc2r bin_id_csr;
    SyncArray<int>& bin_id_val = *bin_id[0];
//    LOG(INFO)<<"bin id"<<bin_id_val;


    int* bin_id_val_data = bin_id_val.device_data();
    int* bin_id_val_host = bin_id_val.host_data();
    LOG(INFO)<<"bin id val:"<<bin_id_val;
    LOG(INFO)<<"csc col:"<<columns.csc_col_ptr;
    float_type* csc_val_device = columns.csc_val.device_data();
    int* csc_row_device = columns.csc_row_ind.device_data();
    int* csc_col_device = columns.csc_col_ptr.device_data();

    float_type* csc_val_host = columns.csc_val.host_data();
    int* csc_row_host = columns.csc_row_ind.host_data();
    int* csc_col_host = columns.csc_col_ptr.host_data();

    SyncArray<float_type> bin_id_float(bin_id_val.size());
    float_type* bin_id_float_data = bin_id_float.device_data();
    device_loop(bin_id_val.size(), [=]__device__(int bid){
        bin_id_float_data[bid] = bin_id_val_data[bid] * 1.0 + 1.0;
    });


    std::cout<<"number of non-zero values before bundle:"<<unsort_columns.nnz<<std::endl;
    LOG(INFO)<<"bin id float"<<bin_id_float;
    LOG(INFO)<<"csc row"<<unsort_columns.csc_row_ind;
    LOG(INFO)<<"csc col"<<unsort_columns.csc_col_ptr;
    bin_id_csr.from_csr(bin_id_float.device_data(), csc_row_device, csc_col_device, n_column, n_instances, columns.nnz);

    LOG(INFO)<<"bin id csr val:"<<bin_id_csr.csc_val;

    float_type* bin_id_csr_val_host = bin_id_csr.csc_val.host_data();
    int* bin_id_csr_row = bin_id_csr.csc_col_ptr.device_data();
    int* bin_id_csr_col = bin_id_csr.csc_row_ind.device_data();
    int* bin_id_csr_row_host = bin_id_csr.csc_col_ptr.host_data();
    int* bin_id_csr_col_host = bin_id_csr.csc_row_ind.host_data();




    //bool* is_del_data = is_del.device_data();
    SyncArray<bool> is_del(n_instances);
    bool* is_del_host = is_del.host_data();



//    SyncArray<int> same_col(n_column);
//    int* same_col_host = same_col.host_data();

    SyncArray<float_type> y_new(n_instances);
    y_new.copy_from(stats_y, n_instances);
    float_type* y_new_host = y_new.host_data();


    vector<vector<float_type>> new_dense_val(n_instances);
    vector<vector<int>> new_csr_col(n_instances);

    vector<int> nnz_perrow(n_instances, 0);
    vector<int> nnz_perrow_diff(numP, 0);
    vector<int> nnz_perrow_diff_current(numP, 0);


    bool* is_multi_data = is_multi.host_data();
    int oiid = 0;
    int miss_col = 0;
    int inequal_dis = 0;
    float dis_percen = 0;

    srand (time(NULL));
    vector<int> instypeid(n_instances, 0); //0 for no similar instance, 1 for front similar, 2 for back similar


    vector<vector<int>> similid(n_instances);

    for(int i = 0; i < n_instances; i++){
        if(is_del_host[i])
            continue;
        iidold2new[i] = oiid;

        int* bin_col_i = bin_id_csr_col_host + bin_id_csr_row_host[i];
        float_type* bin_val_i = bin_id_csr_val_host + bin_id_csr_row_host[i];

        new_dense_val[i].resize(n_column, 0.f);

        for(int t = 0; t < bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]; t++){
            new_dense_val[i][bin_col_i[t]] = bin_val_i[t];

        }


        nnz_perrow[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];
        //nnz_perrow2[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];

        similid[i].push_back(i);

        int nparts = 1;
        bool is_out = 0;
        for(int j = i + 1; j < n_instances; j++){
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) != 0)
//                continue;
            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) > 2)
                continue;
//            if(is_del_host[j])
//                continue;
//            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0))
//                continue;
            float_type y_diff = fabs(stats_y[i] - stats_y[j]);
            int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
            float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//            bool is_same = 1;
            int same_col_num = 0;

            miss_col = 0;
            inequal_dis = 0;

            is_out = 0;
            dis_percen = 0;
            for(int m = 0, n = 0; m <  (bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) && n < (bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]);){
                if(bin_col_i[m] < bin_col_j[n]) {
                    m++;
                    miss_col++;
//                    is_same = 0;
                }
                else if(bin_col_i[m] == bin_col_j[n]){
                    if(bin_val_i[m] != bin_val_j[n]){
//                        is_same = 0;
                        inequal_dis += abs((int)(bin_val_i[m] - bin_val_j[n]));
                        float percen = 1.0 * abs((int)(bin_val_i[m] - bin_val_j[n])) / (bin_id_val_host[csc_col_host[bin_col_i[m] + 1] - 1] + 1);
                        CHECK(percen <= 1.f)<<percen;
                        if(((bin_id_val_host[csc_col_host[bin_col_i[m] + 1] - 1] > 10) && (percen > 0.1))){
                            is_out = 1;
                            break;
                        }
                        dis_percen += percen;
//                        break;
                    }
                    same_col_num ++;
                    m++;
                    n++;
                }
                else {
                    n++;
                    miss_col++;

//                    is_same = 0;
                }
            }
//            if((inequal_dis*5 + miss_col) < 100){
            if(((!is_random) && (is_out == 0) && (dis_percen < 1) && (miss_col <= (n_column / 10))) ||
                (is_random && ((rand() % 100) == 1))){
//            if((rand() % 100) == 1){
//            if(j == (i + n_instances / 2)){

//            if(is_same){

                nparts++;
                iidold2new[j] = oiid;
                is_multi_data[oiid] = 1;
                is_del_host[j] = 1;
                y_new_host[i] = y_new_host[i] + y_new_host[j];
                y_new_host[j] = INFINITY;


                new_dense_val[j].resize(n_column, 0.f);

                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
                    new_dense_val[j][bin_col_j[t]] = bin_val_j[t];

                }

                similid[i].push_back(j);

                nnz_perrow_diff_current[nparts - 1] = bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];
                nnz_perrow_diff[nparts - 1] += bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];

                if(nparts == numP)
                    break;
            }
        }
        if(nparts < numP){
            for(int pid = nparts; pid < numP; pid++){
                similid[i].push_back(similid[i][pid % nparts]);
                nnz_perrow_diff[pid] += nnz_perrow_diff_current[pid % nparts];
            }
        }
        oiid++;
    }


    int del_size = count(is_del.host_data(), is_del.host_data() + n_instances, 1);
    std::cout << "del size:" << del_size << std::endl;
    //del_size = 0;
    if(del_size != 0) {
        int total_nnz_new = reduce(nnz_perrow.begin(), nnz_perrow.begin() + n_instances);

        vector<vector<float_type>> new_csc_val(numP);
        vector<vector<int>> new_csc_row(numP);
        vector<vector<int>> new_csc_col(numP);

#pragma omp parallel for
        for(int i = 0; i < numP; i++){
            new_csc_val[i].resize(total_nnz_new + nnz_perrow_diff[i], 0.f);
            new_csc_row[i].resize(total_nnz_new + nnz_perrow_diff[i]);
            new_csc_col[i].resize(n_column + 1);
            new_csc_col[i][0] = 0;
        }

        vector<vector<float_type>> new_dense_val_total(numP);
//        std::cout<<"vector max size:"<<new_dense_val_total.max_size()<<std::endl;
        int new_row_size = 0;
        for(int i = 0; i < n_instances; i++){
            if(similid[i].size() != 0){
                new_row_size++;
                CHECK(similid[i].size() == numP)<<similid[i].size();
                for(int pid = 0; pid < similid[i].size(); pid++){
                    new_dense_val_total[pid].insert(new_dense_val_total[pid].end(), new_dense_val[similid[i][pid]].begin(), new_dense_val[similid[i][pid]].end());
                }

            }
        }

        vector<int> val_id(numP, 0);
        for(int i = 0; i < n_column; i++){
            for(int j = 0; j < new_row_size; j++){
                for(int pid = 0; pid < numP; pid++){
                    if(new_dense_val_total[pid][j * n_column + i] != 0.f){
                        new_csc_val[pid][val_id[pid]] = new_dense_val_total[pid][j * n_column + i] - 1;
                        new_csc_row[pid][val_id[pid]++] = j;
                    }
                }

            }
            for(int pid = 0; pid < numP; pid++){
                new_csc_col[pid][i + 1] = val_id[pid];
            }
        }


// to do parallel
        for(int pid = 0; pid < numP; pid++){
            (*v_columns[pid][0]).csc_bin_id.resize(new_csc_val[pid].size());
            (*v_columns[pid][0]).csc_bin_id.copy_from(new_csc_val[pid].data(), new_csc_val[pid].size());
            (*v_columns[pid][0]).csc_row_ind.resize(new_csc_row[pid].size());
            (*v_columns[pid][0]).csc_row_ind.copy_from(new_csc_row[pid].data(), new_csc_row[pid].size());
            (*v_columns[pid][0]).csc_col_ptr.resize(new_csc_col[pid].size());
            (*v_columns[pid][0]).csc_col_ptr.copy_from(new_csc_col[pid].data(), new_csc_col[pid].size());
            (*v_columns[pid][0]).nnz = total_nnz_new + nnz_perrow_diff[pid];
            (*v_columns[pid][0]).n_column = n_column;
            cub_seg_sort_by_key((*v_columns[pid][0]).csc_bin_id, (*v_columns[pid][0]).csc_row_ind, (*v_columns[pid][0]).csc_col_ptr, true);
            cudaDeviceSynchronize();
        }

        float_type *y_new_end = remove(y_new_host, y_new_host + n_instances, INFINITY);
        CHECK(del_size == (n_instances - (y_new_end - y_new_host)));
//        std::cout<<"old n_instances: "<<n_instances<<std::endl;
//        std::cout << "del size:" << del_size << std::endl;
        n_instances = n_instances - del_size;
        for(int statid = 0; statid < numP; statid++){
            stats[statid].resize(n_instances);
            stats[statid].y.copy_from(y_new_host, y_new_end - y_new_host);
            if(weighted_gh)
                stats[statid].updateGH(is_multi, numP);
            else
                stats[statid].updateGH();
        }
//        stats.updateGH(is_multi);
//        LOG(INFO)<<"new stat y"<<stats.y;
//        std::cout<<"new stats n_instance: "<<stats.n_instances<<std::endl;

    }
}

void HistUpdater::similar_ins_bundle(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns,
                                            int numP,
                                            vector<InsStat> &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
                                            int* iidold2new, SyncArray<bool>& is_multi){

    using namespace thrust;

    SparseColumns& columns = *v_columns[0][0];

    //SparseColumns& columns = unsort_columns;
    int n_column = columns.n_column;
//    std::cout<<"n_column:"<<n_column;
    float_type *stats_y = stats[0].y.host_data();
    Csc2r bin_id_csr;
    SyncArray<int>& bin_id_val = *bin_id[0];
//    LOG(INFO)<<"bin id"<<bin_id_val;


    int* bin_id_val_data = bin_id_val.device_data();
    float_type* csc_val_device = columns.csc_val.device_data();
    int* csc_row_device = columns.csc_row_ind.device_data();
    int* csc_col_device = columns.csc_col_ptr.device_data();

    float_type* csc_val_host = columns.csc_val.host_data();
    int* csc_row_host = columns.csc_row_ind.host_data();
    int* csc_col_host = columns.csc_col_ptr.host_data();

    SyncArray<float_type> bin_id_float(bin_id_val.size());
    float_type* bin_id_float_data = bin_id_float.device_data();
    device_loop(bin_id_val.size(), [=]__device__(int bid){
        bin_id_float_data[bid] = bin_id_val_data[bid] * 1.0 + 1.0;
    });


    std::cout<<"number of non-zero values before bundle:"<<unsort_columns.nnz<<std::endl;
    LOG(INFO)<<"bin id float"<<bin_id_float;
    LOG(INFO)<<"csc row"<<unsort_columns.csc_row_ind;
    LOG(INFO)<<"csc col"<<unsort_columns.csc_col_ptr;
    bin_id_csr.from_csr(bin_id_float.device_data(), csc_row_device, csc_col_device, n_column, n_instances, columns.nnz);

    LOG(INFO)<<"bin id csr val:"<<bin_id_csr.csc_val;

    float_type* bin_id_csr_val_host = bin_id_csr.csc_val.host_data();
    int* bin_id_csr_row = bin_id_csr.csc_col_ptr.device_data();
    int* bin_id_csr_col = bin_id_csr.csc_row_ind.device_data();
    int* bin_id_csr_row_host = bin_id_csr.csc_col_ptr.host_data();
    int* bin_id_csr_col_host = bin_id_csr.csc_row_ind.host_data();




    //bool* is_del_data = is_del.device_data();
    SyncArray<bool> is_del(n_instances);
    bool* is_del_host = is_del.host_data();



//    SyncArray<int> same_col(n_column);
//    int* same_col_host = same_col.host_data();

    SyncArray<float_type> y_new(n_instances);
    y_new.copy_from(stats_y, n_instances);
    float_type* y_new_host = y_new.host_data();


    vector<vector<float_type>> new_dense_val(n_instances);
    vector<vector<int>> new_csr_col(n_instances);

    vector<int> nnz_perrow(n_instances, 0);
    vector<int> nnz_perrow_diff(numP, 0);
    vector<int> nnz_perrow_diff_current(numP, 0);


    bool* is_multi_data = is_multi.host_data();
    int oiid = 0;
    int miss_col = 0;
    int inequal_dis = 0;

    srand (time(NULL));
    vector<int> instypeid(n_instances, 0); //0 for no similar instance, 1 for front similar, 2 for back similar


    vector<vector<int>> similid(n_instances);

    for(int i = 0; i < n_instances; i++){
        if(is_del_host[i])
            continue;
        iidold2new[i] = oiid;

        int* bin_col_i = bin_id_csr_col_host + bin_id_csr_row_host[i];
        float_type* bin_val_i = bin_id_csr_val_host + bin_id_csr_row_host[i];

        new_dense_val[i].resize(n_column, 0.f);

        for(int t = 0; t < bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]; t++){
            new_dense_val[i][bin_col_i[t]] = bin_val_i[t];

        }


        nnz_perrow[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];
        //nnz_perrow2[i] = bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i];

        similid[i].push_back(i);

        int nparts = 1;
//        int small_dis = 100;
//        int small_ind;
        for(int j = i + 1; j < n_instances; j++){
            //            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) != 0)
            //                continue;
            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0) || fabs(stats_y[i] - stats_y[j]) > 2)
                continue;
            //            if(is_del_host[j] || (stats_y[i] * stats_y[j] < 0))
            //                continue;
            float_type y_diff = fabs(stats_y[i] - stats_y[j]);
            int* bin_col_j = bin_id_csr_col_host + bin_id_csr_row_host[j];
            float_type* bin_val_j = bin_id_csr_val_host + bin_id_csr_row_host[j];
//            bool is_same = 1;
            int same_col_num = 0;

            miss_col = 0;
            inequal_dis = 0;



            for(int m = 0, n = 0; m <  (bin_id_csr_row_host[i + 1] - bin_id_csr_row_host[i]) && n < (bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]);){
                if(bin_col_i[m] < bin_col_j[n]) {
                    m++;
                    miss_col++;
                }
                else if(bin_col_i[m] == bin_col_j[n]){
                    if(bin_val_i[m] != bin_val_j[n]){
//                        is_same = 0;
                        inequal_dis += abs((int)(bin_val_i[m] - bin_val_j[n]));
                    }
                    same_col_num ++;
                    m++;
                    n++;
                }
                else {
                    n++;
                    miss_col++;
                }
            }
            if((inequal_dis*5 + miss_col) < 100){
            //            if((rand() % 100) == 1){
            //            if(j == (i + n_instances / 2)){

            //            if(is_same){

                nparts++;
                iidold2new[j] = oiid;
                is_multi_data[oiid] = 1;
                is_del_host[j] = 1;
                y_new_host[i] = y_new_host[i] + y_new_host[j];
                y_new_host[j] = INFINITY;


                new_dense_val[j].resize(n_column, 0.f);

                for(int t = 0; t < bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j]; t++){
                    new_dense_val[j][bin_col_j[t]] = bin_val_j[t];
                }

                similid[i].push_back(j);

                nnz_perrow_diff_current[nparts - 1] = bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];
                nnz_perrow_diff[nparts - 1] += bin_id_csr_row_host[j + 1] - bin_id_csr_row_host[j] - nnz_perrow[i];

                if(nparts == numP)
                    break;
            }
        }

        if(nparts < numP){
            for(int pid = nparts; pid < numP; pid++){
                similid[i].push_back(similid[i][pid % nparts]);
                nnz_perrow_diff[pid] += nnz_perrow_diff_current[pid % nparts];
            }
        }
        oiid++;
    }


    int del_size = count(is_del.host_data(), is_del.host_data() + n_instances, 1);
    std::cout << "del size:" << del_size << std::endl;
    //del_size = 0;
    if(del_size != 0) {
        int total_nnz_new = reduce(nnz_perrow.begin(), nnz_perrow.begin() + n_instances);

        vector<vector<float_type>> new_csc_val(numP);
        vector<vector<int>> new_csc_row(numP);
        vector<vector<int>> new_csc_col(numP);

#pragma omp parallel for
        for(int i = 0; i < numP; i++){
            new_csc_val[i].resize(total_nnz_new + nnz_perrow_diff[i], 0.f);
            new_csc_row[i].resize(total_nnz_new + nnz_perrow_diff[i]);
            new_csc_col[i].resize(n_column + 1);
            new_csc_col[i][0] = 0;
        }

        vector<vector<float_type>> new_dense_val_total(numP);
        int new_row_size = 0;
        for(int i = 0; i < n_instances; i++){
            if(similid[i].size() != 0){
                new_row_size++;
                CHECK(similid[i].size() == numP)<<similid[i].size();
                for(int pid = 0; pid < similid[i].size(); pid++){
                    new_dense_val_total[pid].insert(new_dense_val_total[pid].end(), new_dense_val[similid[i][pid]].begin(), new_dense_val[similid[i][pid]].end());
                }

            }
        }

        vector<int> val_id(numP, 0);
        for(int i = 0; i < n_column; i++){
            for(int j = 0; j < new_row_size; j++){
                for(int pid = 0; pid < numP; pid++){
                    if(new_dense_val_total[pid][j * n_column + i] != 0.f){
                        new_csc_val[pid][val_id[pid]] = new_dense_val_total[pid][j * n_column + i] - 1;
                        new_csc_row[pid][val_id[pid]++] = j;
                    }
                }

            }
            for(int pid = 0; pid < numP; pid++){
                new_csc_col[pid][i + 1] = val_id[pid];
            }
        }


// to do parallel
        for(int pid = 0; pid < numP; pid++){
            (*v_columns[pid][0]).csc_bin_id.resize(new_csc_val[pid].size());
            (*v_columns[pid][0]).csc_bin_id.copy_from(new_csc_val[pid].data(), new_csc_val[pid].size());
            (*v_columns[pid][0]).csc_row_ind.resize(new_csc_row[pid].size());
            (*v_columns[pid][0]).csc_row_ind.copy_from(new_csc_row[pid].data(), new_csc_row[pid].size());
            (*v_columns[pid][0]).csc_col_ptr.resize(new_csc_col[pid].size());
            (*v_columns[pid][0]).csc_col_ptr.copy_from(new_csc_col[pid].data(), new_csc_col[pid].size());
            (*v_columns[pid][0]).nnz = total_nnz_new + nnz_perrow_diff[pid];
            (*v_columns[pid][0]).n_column = n_column;
            cub_seg_sort_by_key((*v_columns[pid][0]).csc_bin_id, (*v_columns[pid][0]).csc_row_ind, (*v_columns[pid][0]).csc_col_ptr, true);
            cudaDeviceSynchronize();
        }

        float_type *y_new_end = remove(y_new_host, y_new_host + n_instances, INFINITY);
        CHECK(del_size == (n_instances - (y_new_end - y_new_host)));
//        std::cout<<"old n_instances: "<<n_instances<<std::endl;
//        std::cout << "del size:" << del_size << std::endl;
        n_instances = n_instances - del_size;
//        std::cout<<"new n_instances:"<<n_instances<<std::endl;
        for(int statid = 0; statid < numP; statid++){
            stats[statid].resize(n_instances);
            stats[statid].y.copy_from(y_new_host, y_new_end - y_new_host);
            stats[statid].updateGH();

//            stats[statid].updateGH(is_multi, numP);

        }
//        stats.updateGH(is_multi);
//        LOG(INFO)<<"new stat y"<<stats.y;
//        std::cout<<"new stats n_instance: "<<stats.n_instances<<std::endl;

    }
}

void HistUpdater::find_split(int level, const SparseColumns &columns, const Tree &tree, const InsStat &stats,
                             SyncArray<SplitPoint> &sp) {
    TIMED_SCOPE(timerObj, "find split inside");
//    TIMED_SCOPE(timerObj, "init");
    int n_max_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = columns.n_column;
    int n_partition = n_column * n_max_nodes_in_level;
    int nnz = columns.nnz;
    int n_block = std::min((nnz / n_column - 1) / 256 + 1, 32 * 56);

    int cur_device;
    cudaGetDevice(&cur_device);
    LOG(TRACE) << "start finding split";

    //find the best split locally
    {
        TIMED_SCOPE(timerObj, "in big find");
        using namespace thrust;
        SyncArray<int> fvid2pid(nnz);

        {
            TIMED_SCOPE(timerObj, "fvid2pid");
            //input
            const int *nid_data = v_stats[cur_device]->nid.device_data();//######### change to vector
            const int *iid_data = columns.csc_row_ind.device_data();

            LOG(TRACE) << "after using v_stats and columns";
            //output
            int *fvid2pid_data = fvid2pid.device_data();
            device_loop_2d(
                    n_column, columns.csc_col_ptr.device_data(),
                    [=]__device__(int col_id, int fvid) {
                        //feature value id -> instance id -> node id
                        int nid = nid_data[iid_data[fvid]];
                        int pid;
                        //if this node is leaf node, move it to the end
                        if (nid < nid_offset) pid = INT_MAX;//todo negative
                        else pid = (nid - nid_offset) * n_column + col_id;
                        fvid2pid_data[fvid] = pid;
                    },
                    n_block);
            cudaDeviceSynchronize();
            LOG(DEBUG) << "fvid2pid " << fvid2pid;
        }




        //gather g/h pairs and do prefix sum
        int n_split;
        SyncArray <GHPair> gh_prefix_sum;
        SyncArray <GHPair> missing_gh(n_partition);

        //SyncArray<float_type> rle_fval;
        SyncArray<int> rle_pid;

        int cut_points_size = v_cut[cur_device].cut_points_val.size();
        SyncArray<int> cut_bin_id(cut_points_size);
        auto cut_bin_id_ptr = cut_bin_id.device_data();
        //bool do_exact = true;

        auto bin_id_ptr = columns.csc_bin_id.device_data();

        {
            //get feature value id mapping for partition, new -> old
            SyncArray<int> fvid_new2old(nnz);
            {
                TIMED_SCOPE(timerObj, "fvid_new2old");
                sequence(cuda::par, fvid_new2old.device_data(), fvid_new2old.device_end(), 0);
                stable_sort_by_key(
                        cuda::par, fvid2pid.device_data(), fvid2pid.device_end(),
                        fvid_new2old.device_data(),
                        thrust::less<int>());
                LOG(DEBUG) << "sorted fvid2pid " << fvid2pid;
                LOG(DEBUG) << "fvid_new2old " << fvid_new2old;
            }
            {
                TIMED_SCOPE(timerObj, "hist");
                n_split = n_partition / n_column * v_cut[cur_device].cut_points.size();
                gh_prefix_sum.resize(n_split);
                rle_pid.resize(n_split);
                //rle_fval.resize(n_split);


                SyncArray<int> p_size_prefix_sum(n_partition + 1);
                //p_size_prefix_sum.host_data()[0] = 0;
                counting_iterator<int> search_begin(0);
                upper_bound(cuda::par, fvid2pid.device_data(), fvid2pid.device_end(), search_begin,
                            search_begin + n_partition, p_size_prefix_sum.device_data() + 1);
                LOG(TRACE) << "p_size_prefix_sum:" << p_size_prefix_sum;
                int n_f = 0;

                cudaMemcpy(&n_f, p_size_prefix_sum.device_data() + n_partition, sizeof(int),
                           cudaMemcpyDeviceToHost);

//                auto fval_iter = make_permutation_iterator(columns.csc_val.device_data(),
//                                                           fvid_new2old.device_data());
//                auto fval_iter = make_permutation_iterator(columns.csc_bin_id.device_data(),
//                                                           fvid_new2old.device_data());


                LOG(TRACE) << "fvid new2old:" << fvid_new2old;
                auto iid_iter = make_permutation_iterator(
                        columns.csc_row_ind.device_data(), fvid_new2old.device_data());
                auto p_size_prefix_sum_ptr = p_size_prefix_sum.device_data();

                auto cut_points_ptr = v_cut[cur_device].cut_points_val.device_data();
                auto gh_prefix_sum_ptr = gh_prefix_sum.device_data();

                auto cut_row_ptr = v_cut[cur_device].cut_row_ptr.device_data();
                auto stats_gh_ptr = v_stats[cur_device]->gh_pair.device_data();

                SyncArray<int> cut_off(n_f);
                auto cut_off_ptr = cut_off.device_data();

//                auto bin_iter = make_permutation_iterator((*bin_id[cur_device]).device_data(),
//                        fvid_new2old.device_data());

                auto bin_iter = make_permutation_iterator(columns.csc_bin_id.device_data(),
                                                          fvid_new2old.device_data());
                //LOG(INFO)<<"find split bin id"<<*bin_id[cur_device];
                copy(cuda::par, bin_iter, bin_iter + n_f, cut_off_ptr);
                device_loop(n_partition, [=]__device__(int pid){
                    int cut_start = pid / n_column * cut_points_size + cut_row_ptr[pid % n_column];
                    for_each(cuda::par, cut_off_ptr + p_size_prefix_sum_ptr[pid],
                             cut_off_ptr + p_size_prefix_sum_ptr[pid + 1],
                             thrust::placeholders::_1 += cut_start);
                });

//                device_loop(n_partition,
//                            [=]__device__(int pid) {
//                                int partition_size = p_size_prefix_sum_ptr[pid + 1] - p_size_prefix_sum_ptr[pid];
//                                auto partition_start = fval_iter + p_size_prefix_sum_ptr[pid];
//                                //auto iid = iid_iter + p_size_prefix_sum_ptr[pid];
//                                auto cbegin = cut_points_ptr + cut_row_ptr[pid % n_column];
//                                auto cend = cut_points_ptr + cut_row_ptr[pid % n_column + 1];
//                                //int bin_size = cend - cbegin;
//                                lower_bound(cuda::par, cbegin, cend, partition_start,
//                                            partition_start + partition_size,
//                                            cut_off_ptr + p_size_prefix_sum_ptr[pid],
//                                            thrust::greater<float_type>());
//                                //replace(cuda::par, cut_off_ptr + p_size_prefix_sum_ptr[pid], cut_off_ptr + p_size_prefix_sum_ptr[pid + 1], bin_size, bin_size - 1);
//
//                                int cut_start = pid / n_column * cut_points_size + cut_row_ptr[pid % n_column];
//                                for_each(cuda::par, cut_off_ptr + p_size_prefix_sum_ptr[pid],
//                                         cut_off_ptr + p_size_prefix_sum_ptr[pid + 1],
//                                         thrust::placeholders::_1 += cut_start);
//
//                            });

                auto gh_insid_ptr = make_permutation_iterator(stats_gh_ptr, iid_iter);
                SyncArray <GHPair> gh_ins(n_f);
                auto gh_ins_ptr = gh_ins.device_data();
                copy(cuda::par, gh_insid_ptr, gh_insid_ptr + n_f, gh_ins_ptr);

                //cudaDeviceSynchronize();

                //LOG(INFO)<<"cut_off "<<cut_off;
                stable_sort_by_key(cuda::par, cut_off_ptr, cut_off_ptr + n_f, gh_ins_ptr);

                SyncArray<int> cut_off_after_reduce(n_split);
                auto cut_off_after_reduce_ptr = cut_off_after_reduce.device_data();

                int n_bin = reduce_by_key(cuda::par, cut_off_ptr, cut_off_ptr + n_f, gh_ins_ptr,
                                          cut_off_after_reduce_ptr, gh_ins_ptr).first -
                            cut_off_after_reduce.device_data();
                LOG(TRACE) << "cut_off_after reduce" << cut_off_after_reduce;

                device_loop(n_bin, [=]__device__(int i) {
                    gh_prefix_sum_ptr[cut_off_after_reduce_ptr[i]].g = gh_ins_ptr[i].g;
                    gh_prefix_sum_ptr[cut_off_after_reduce_ptr[i]].h = gh_ins_ptr[i].h;
                });

//                auto rle_fval_ptr = rle_fval.device_data();
//                device_loop(n_split, [=]__device__(int i) {
//                    rle_fval_ptr[i] = cut_points_ptr[i % cut_points_size];
//                });



                device_loop(n_column, [=]__device__(int i){
                    sequence(cuda::par, cut_bin_id_ptr + cut_row_ptr[i], cut_bin_id_ptr + cut_row_ptr[i + 1]);
                });


                auto rle_pid_ptr = rle_pid.device_data();

                device_loop_2d_mod(n_partition, n_column, cut_row_ptr,
                                   [=]
                __device__(int
                pid, int
                cut_off) {
                    int off = pid / n_column * cut_points_size + cut_off;
                    rle_pid_ptr[off] = pid;
                }, 1);

                inclusive_scan_by_key(
                        cuda::par,
                        rle_pid.device_data(), rle_pid.device_end(),
                        gh_prefix_sum.device_data(),
                        gh_prefix_sum.device_data());


                //const auto gh_prefix_sum_ptr = gh_prefix_sum.device_data();
                const auto node_ptr = tree.nodes.device_data();
                auto missing_gh_ptr = missing_gh.device_data();
                auto cut_row_ptr_device = v_cut[cur_device].cut_row_ptr.device_data();

                device_loop(n_partition, [=]__device__(int pid) {
                    int nid = pid / n_column + nid_offset;
                    int off = pid / n_column * cut_points_size + cut_row_ptr_device[pid % n_column + 1] - 1;
                    if (p_size_prefix_sum_ptr[pid + 1] != p_size_prefix_sum_ptr[pid])
                        missing_gh_ptr[pid] =
                                node_ptr[nid].sum_gh_pair - gh_prefix_sum_ptr[off];
                });

                cudaDeviceSynchronize();
            }
        }


        //calculate gain of each split
        SyncArray<float_type> gain(n_split);
        SyncArray<bool> default_right(n_split);
        {
            TIMED_SCOPE(timerObj, "calculate gain");
            auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                             float_type lambda) -> float_type {
                if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                    return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
                           (father.g * father.g) / (father.h + lambda);
                else
                    return 0;
            };

            int *fvid2pid_data = fvid2pid.device_data();
            const Tree::TreeNode *nodes_data = tree.nodes.device_data();
//                float_type *f_val_data = columns.csc_val.device_data();
            GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
            float_type *gain_data = gain.device_data();
            bool *default_right_data = default_right.device_data();
            const auto rle_pid_data = rle_pid.device_data();
            const auto missing_gh_data = missing_gh.device_data();
            //for lambda expression
            float_type mcw = min_child_weight;
            float_type l = lambda;
            device_loop(n_split, [=]__device__(int i) {
                int pid = rle_pid_data[i];
                int nid0 = pid / n_column;
                int nid = nid0 + nid_offset;
                if (pid == INT_MAX) return;
                GHPair father_gh = nodes_data[nid].sum_gh_pair;
                GHPair p_missing_gh = missing_gh_data[pid];
                GHPair rch_gh = gh_prefix_sum_data[i];
                float_type max_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                if (p_missing_gh.h > 1) {
                    rch_gh = rch_gh + p_missing_gh;
                    float_type temp_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                    if (temp_gain > 0 && temp_gain - max_gain > 0.1) {
                        max_gain = temp_gain;
                        default_right_data[i] = true;
                    }
                }
                gain_data[i] = max_gain;
            });
            cudaDeviceSynchronize();
            LOG(DEBUG) << "gain = " << gain;
        }

        //get best gain and the index of best gain for each feature and each node
        SyncArray<int_float> best_idx_gain(n_max_nodes_in_level);
        int n_nodes_in_level;
        {
            TIMED_SCOPE(timerObj, "get best gain");
            auto arg_max = []__device__(const int_float &a, const int_float &b) {
                if (get<1>(a) == get<1>(b))
                    return get<0>(a) < get<0>(b) ? a : b;
                else
                    return get<1>(a) > get<1>(b) ? a : b;
            };
            auto in_same_node = [=]__device__(const int a, const int b) {
                return (a / n_column) == (b / n_column);
            };

            //reduce to get best split of each node for this feature
            SyncArray<int> key_test(n_max_nodes_in_level);
            n_nodes_in_level = reduce_by_key(
                    cuda::par,
                    rle_pid.device_data(), rle_pid.device_end(),
                    make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.device_data())),
                    key_test.device_data(),//make_discard_iterator(),
                    best_idx_gain.device_data(),
                    in_same_node,
                    arg_max).second - best_idx_gain.device_data();

            LOG(DEBUG) << "#nodes in level = " << n_nodes_in_level;
            LOG(DEBUG) << "best pid = " << key_test;
            LOG(DEBUG) << "best idx & gain = " << best_idx_gain;
        }

        {
            TIMED_SCOPE(timerObj, "get split point");
            //get split points
            const int_float *best_idx_gain_data = best_idx_gain.device_data();
            const auto rle_pid_data = rle_pid.device_data();
            //Tree::TreeNode *nodes_data = tree.nodes.device_data();
            GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
            //const auto rle_fval_data = rle_fval.device_data();
            const auto missing_gh_data = missing_gh.device_data();
            bool *default_right_data = default_right.device_data();

            sp.resize(n_nodes_in_level);
            auto sp_data = sp.device_data();

            int column_offset = columns.column_offset;
            device_loop(n_nodes_in_level, [=]__device__(int i) {
                int_float bst = best_idx_gain_data[i];
                float_type best_split_gain = get<1>(bst);
                int split_index = get<0>(bst);
                int pid = rle_pid_data[split_index];
                sp_data[i].split_fea_id = (pid == INT_MAX) ? -1 : (pid % n_column) + column_offset;
                sp_data[i].nid = (pid == INT_MAX) ? -1 : (pid / n_column + nid_offset);
                sp_data[i].gain = best_split_gain;
                if (pid != INT_MAX) {//avoid split_index out of bound
//                sp_data[i].fval = rle_fval_data[split_index];
                    sp_data[i].fval = (float_type)cut_bin_id_ptr[split_index % cut_points_size];
                    sp_data[i].fea_missing_gh = missing_gh_data[pid];
                    sp_data[i].default_right = default_right_data[split_index];
                    sp_data[i].rch_sum_gh = gh_prefix_sum_data[split_index];
                }
            });
        }

    }

    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}


bool HistUpdater::reset_ins2node_id(InsStat &stats, const Tree &tree, const SparseColumns &columns) {
    SyncArray<bool> has_splittable(1);
    //set new node id for each instance
    {
        TIMED_SCOPE(timerObj, "get new node id");
        int *nid_data = stats.nid.device_data();
        const int *iid_data = columns.csc_row_ind.device_data();
        const Tree::TreeNode *nodes_data = tree.nodes.device_data();
        const int *col_ptr_data = columns.csc_col_ptr.device_data();
        //const float_type *f_val_data = columns.csc_val.device_data();
        const float_type *bin_id_data = columns.csc_bin_id.device_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.device_data();
        int column_offset = columns.column_offset;

        int n_column = columns.n_column;
        int nnz = columns.nnz;
        int n_block = std::min((nnz / n_column - 1) / 256 + 1, 32 * 56);

        LOG(TRACE) << "update ins2node id for each fval";
        device_loop_2d(n_column, col_ptr_data,
                       [=]__device__(int col_id, int fvid) {
            //feature value id -> instance id
            int iid = iid_data[fvid];
            //instance id -> node id
            int nid = nid_data[iid];
            //node id -> node
            const Tree::TreeNode &node = nodes_data[nid];
            //if the node splits on this feature
            if (node.splittable() && node.split_feature_id == col_id + column_offset) {
                h_s_data[0] = true;
                //if (f_val_data[fvid] < node.split_value)
                if (bin_id_data[fvid] > node.split_value)
                    //goes to left child
                    nid_data[iid] = node.lch_index;
                else
                    //right child
                    nid_data[iid] = node.rch_index;
            }
        }, n_block);

    }
    LOG(DEBUG) << "new tree_id = " << stats.nid;
//        LOG(DEBUG) << v_trees_gpu[cur_device_id].nodes;
    return has_splittable.host_data()[0];
}
