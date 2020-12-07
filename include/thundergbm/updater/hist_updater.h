//
// Created by qinbin on 2018/7/6.
//

#ifndef GBM_MIRROR2_HIST_UPDATER_H
#define GBM_MIRROR2_HIST_UPDATER_H

#include "thundergbm/updater/exact_updater.h"
#include "thundergbm/hist_cut.h"
#include "thundergbm/csc2r_transform.h"

class HistUpdater : public ExactUpdater{
public:
    int max_num_bin = 64;
    int do_cut = 0;
    bool use_similar_bundle = 1;
    vector<HistCut> v_cut;
    vector<std::shared_ptr<SyncArray<int>>> bin_id;


    void insBundle(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats);
    void init_bin_id(const vector<std::shared_ptr<SparseColumns>> &v_columns);
    void init_bin_id_outside(const vector<std::shared_ptr<SparseColumns>> &v_columns, SyncArray<int>& bin_id);
    void init_bin_id_unsort(SparseColumns& unsort_columns, SyncArray<int>& bin_id);
    void copy_bin_id(const vector<std::shared_ptr<SparseColumns>> &v_columns, SyncArray<int>& bin_id);
    void init_cut(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats, int n_instance,
            SparseColumns& unsort_columns);

    void init_bin_id_csr(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns, int n_instances);

    void similar_ins_bundle(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats,
            int& n_instances, DataSet &dataSet, SparseColumns& unsort_columns, int* iidold2new, SyncArray<bool>& is_multi);
    void similar_ins_bundle(const vector<std::shared_ptr<SparseColumns>> &v_columns,
            const vector<std::shared_ptr<SparseColumns>> &v_columns2, InsStat &stats,
                            int& n_instances, DataSet &dataSet, SparseColumns& unsort_columns, int* iidold2new, SyncArray<bool>& is_multi);
    void similar_ins_bundle_multi(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns,
                            int numP, InsStat &stats, int& n_instances, DataSet &dataSet,
                            SparseColumns& unsort_columns, int* iidold2new, SyncArray<bool>& is_multi, bool is_random = 0);
    void similar_ins_bundle_closest(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns, int numP,
            InsStat &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,int* iidold2new, SyncArray<bool>& is_multi);
    void similar_ins_bundle(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns, int numP,
            vector<InsStat> &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
            int* iidold2new, SyncArray<bool>& is_multi);


    void similar_ins_bundle_independent(const vector<vector<std::shared_ptr<SparseColumns>>> &v_columns, int numP,
            vector<InsStat> &stats, int& n_instances, DataSet& dataSet, SparseColumns& unsort_columns,
            int* iidold2new, SyncArray<bool>& is_multi, bool is_random = 0, bool weighted_gh = 1);

    void get_bin_ids(const SparseColumns &columns);

    void find_split(int level, const SparseColumns &columns, const Tree &tree, const InsStat &stats,
                    SyncArray<SplitPoint> &sp) override;

    bool reset_ins2node_id(InsStat &stats, const Tree &tree, const SparseColumns &columns) override;

    explicit HistUpdater(GBMParam &param): ExactUpdater(param) {};



    Csc2r bin_id_csr;

};
#endif //GBM_MIRROR2_HIST_UPDATER_H
