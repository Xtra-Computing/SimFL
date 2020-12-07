//
// Created by jiashuai on 18-1-18.
//
#include <thundergbm/tree.h>
#include <thundergbm/dataset.h>
#include <thundergbm/updater/exact_updater.h>
#include <thundergbm/updater/hist_updater.h>
#include <thundergbm/util/cub_wrapper.h>
#include "thundergbm/util/device_lambda.cuh"
//#include <thundergbm/gpu_lsh.h>
#include "gtest/gtest.h"
#include <ctime>
#include <cstdlib>
#include <fstream>
extern int iargc;
extern char** iargv;

string dataset_name = "a9a_dir/";
int n_table=20;
int n_bucket=50;
float r=4.0;
int numP=2;
int max_dim=0;
string file_name="text.txt";
string dataset_path = DATASET_DIR + dataset_name;
int lsh_seed=-1;

int gpu_id = 0;
int n_trees = 50;
int n_depth = 8;
class UpdaterTest : public ::testing::Test {
public:

    GBMParam param;
    bool verbose = false;

    void SetUp() override {
        //common param
        param.depth = 8;
        param.n_trees = 50;
        param.min_child_weight = 1;
        param.lambda = 1;
        param.gamma = 1;
        param.rt_eps = 1e-6;
        param.do_exact = true;
        param.n_device = 1;
        param.learning_rate = 0.1;

        if (!verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "true");

    }

    float_type train_exact(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);
        int n_instances = dataSet.n_instances();
        InsStat stats;
        vector<Tree> trees;
        SparseColumns columns;
        columns.from_dataset(dataSet);
        trees.resize(param.n_trees);
        stats.resize(n_instances);
        stats.y.copy_from(dataSet.y().data(), n_instances);
        int* nid_data = stats.nid.host_data();

        int n_devices = param.n_device;
        vector<std::shared_ptr<SparseColumns>> v_columns;
        v_columns.resize(n_devices);
        for (int i = 0; i < n_devices; i++)
            v_columns[i].reset(new SparseColumns());
        columns.to_multi_devices(v_columns);
        ExactUpdater updater(param);
        int round = 0;
        float_type rmse = 0;
        {
            TIMED_SCOPE(timerObj, "construct tree");
            for (Tree &tree:trees) {
                TIMED_SCOPE(timerObj, "ExactEachTree");
                stats.updateGH();
                updater.grow(tree, v_columns, stats);
                tree.prune_self(param.gamma);
                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                predict_in_training(stats, tree);
                //next round
                round++;
            }
        }

        DataSet test_dataSet;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_exact(trees, test_dataSet, updater, y_predict);

        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;
        }

        float_type acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error = "<< (1 - acc);


        float_type sum_error = 0;
        for (int i = 0; i < test_instances; i++) {
            float_type e = y_predict[i] - test_dataSet.y()[i];
            sum_error += e * e;
        }
        float_type test_rmse = sqrt(sum_error / test_instances);
        LOG(INFO)<<"test rmse = "<<test_rmse;

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"train with exact:"<<(1-acc)<<std::endl;
        outfile.close();
        return rmse;
    }

    float_type predict_test_dataset_rmse_exact(const vector<Tree> &trees, DataSet& dataSet, ExactUpdater& updater, vector<float_type>& y_predict){
        size_t n_instances = dataSet.n_instances();

        SparseColumns columns;

        columns.from_dataset(dataSet);
        int n_column = columns.n_column;
        const int *iid_data = columns.csc_row_ind.host_data();
        const int *col_ptr_data = columns.csc_col_ptr.host_data();
        const float_type *csc_val_data = columns.csc_val.host_data();
        vector<int> nid_data(n_instances, 0);
        vector<int> nid_data_update(n_instances, 0);

        for(int tid = 0; tid < trees.size(); tid++) {
            const Tree &tree = trees[tid];
            const Tree::TreeNode *nodes_data = tree.nodes.host_data();
            for(int i = 0; i < n_instances; i++)
                nid_data[i] = 0;
            for(int depth = 0; depth < updater.depth; depth++) {
                int n_max_nodes_in_level = 1 << depth;//2^i
                int nid_offset = (1 << depth) - 1;//2^i - 1
                for(int iid = 0; iid < n_instances; iid++)
                    nid_data_update[iid] = nid_data[iid];
//#pragma omp parallel for
                for (int col_id = 0; col_id < n_column; col_id++) {
                    for (int fid = col_ptr_data[col_id]; fid < col_ptr_data[col_id + 1]; fid++) {


                        int iid = iid_data[fid];
                        int nid = nid_data[iid];

                        const Tree::TreeNode &node = nodes_data[nid];
                        if ((node.splittable()) && (node.split_feature_id == col_id)) {
                            if (csc_val_data[fid] < node.split_value) {
                                //goes to left child
                                nid_data_update[iid] = node.lch_index;
                            }
                            else {
                                //right child
                                nid_data_update[iid] = node.rch_index;
                            }
                        }
                    }
                }

                for(int iid = 0; iid < n_instances; iid++) {
                    nid_data[iid] = nid_data_update[iid];
                }
                for(int iid = 0; iid < n_instances; iid++){
                    int nid = nid_data[iid];
                    if(nodes_data[nid].splittable() && (nid < nid_offset + n_max_nodes_in_level)) {
                        const Tree::TreeNode &node = nodes_data[nid];
                        if (node.default_right)
                            nid_data[iid] = node.rch_index;
                        else
                            nid_data[iid] = node.lch_index;
                    }
                }
            }

            for(int iid = 0; iid < n_instances; iid++) {
                int nid = nid_data[iid];
                while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
                y_predict[iid] += nodes_data[nid].base_weight;
            }
            float_type sum_error = 0;

            for(int i = 0; i < n_instances; i++){
                float_type e = y_predict[i] - dataSet.y()[i];
                sum_error += e * e;
            }
            float_type rmse = sqrt(sum_error / n_instances);
        }
        float_type sum_error = 0;

        for(int i = 0; i < n_instances; i++){
            float_type e = y_predict[i] - dataSet.y()[i];
            sum_error += e * e;
        }
        float_type rmse = sqrt(sum_error / n_instances);

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"rmse: "<< rmse<<std::endl;
        outfile.close();

        return rmse;

    }


    float_type predict_test_dataset_rmse_exact(const vector<Tree> &trees, DataSet& dataSet, ExactUpdater& updater, vector<float_type>& y_predict, int n_tree){
        size_t n_instances = dataSet.n_instances();
        SparseColumns columns;

        columns.from_dataset(dataSet);
        int n_column = columns.n_column;
        const int *iid_data = columns.csc_row_ind.host_data();
        const int *col_ptr_data = columns.csc_col_ptr.host_data();
        const float_type *csc_val_data = columns.csc_val.host_data();
        vector<int> nid_data(n_instances, 0);
        vector<int> nid_data_update(n_instances, 0);
        for(int tid = 0; tid < n_tree; tid++) {
            const Tree &tree = trees[tid];
            const Tree::TreeNode *nodes_data = tree.nodes.host_data();
            for(int i = 0; i < n_instances; i++)
                nid_data[i] = 0;
            for(int depth = 0; depth < updater.depth; depth++) {
                int n_max_nodes_in_level = 1 << depth;//2^i
                int nid_offset = (1 << depth) - 1;//2^i - 1
                for(int iid = 0; iid < n_instances; iid++)
                    nid_data_update[iid] = nid_data[iid];
                for (int col_id = 0; col_id < n_column; col_id++) {
                    for (int fid = col_ptr_data[col_id]; fid < col_ptr_data[col_id + 1]; fid++) {


                        int iid = iid_data[fid];
                        int nid = nid_data[iid];

                        const Tree::TreeNode &node = nodes_data[nid];
                        if ((node.splittable()) && (node.split_feature_id == col_id)) {
                            if (csc_val_data[fid] < node.split_value) {
                                //goes to left child
                                nid_data_update[iid] = node.lch_index;
                            }
                            else {
                                //right child
                                nid_data_update[iid] = node.rch_index;
                            }
                        }
                    }
                }

                for(int iid = 0; iid < n_instances; iid++) {
                    nid_data[iid] = nid_data_update[iid];
                }
                for(int iid = 0; iid < n_instances; iid++){
                    int nid = nid_data[iid];
                    if(nodes_data[nid].splittable() && (nid < nid_offset + n_max_nodes_in_level)) {
                        const Tree::TreeNode &node = nodes_data[nid];
                        if (node.default_right)
                            nid_data[iid] = node.rch_index;
                        else
                            nid_data[iid] = node.lch_index;
                    }
                }
            }

            for(int iid = 0; iid < n_instances; iid++) {
                int nid = nid_data[iid];
                while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
                y_predict[iid] += nodes_data[nid].base_weight;
            }
            float_type sum_error = 0;

            for(int i = 0; i < n_instances; i++){
                float_type e = y_predict[i] - dataSet.y()[i];
                sum_error += e * e;
            }
            float_type rmse = sqrt(sum_error / n_instances);
        }
        float_type sum_error = 0;

        for(int i = 0; i < n_instances; i++){
            float_type e = y_predict[i] - dataSet.y()[i];
            sum_error += e * e;
        }
        float_type rmse = sqrt(sum_error / n_instances);

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"rmse: "<< rmse<<std::endl;
        outfile.close();

        return rmse;

    }

    float_type train_hist(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);


        int n_instances = dataSet.n_instances();
        LOG(INFO)<<"before iidold2new";
        int old_instances = n_instances;
        int* iidold2new = new int [old_instances];
        LOG(INFO)<<"after iidold2new";
        thrust::sequence(iidold2new, iidold2new + old_instances);
        SyncArray<bool> is_multi(n_instances);

        InsStat stats;
        vector<Tree> trees;
        SparseColumns columns;
        columns.from_dataset(dataSet);
        trees.resize(param.n_trees);
        LOG(INFO)<<"before stats";
        stats.resize(n_instances);
        stats.y.copy_from(dataSet.y().data(), n_instances);
        LOG(INFO)<<"after stats";
        int n_devices = 1;
        vector<std::shared_ptr<SparseColumns>> v_columns;
        v_columns.resize(n_devices);
        for (int i = 0; i < n_devices; i++)
            v_columns[i].reset(new SparseColumns());
        columns.to_multi_devices(v_columns);
        HistUpdater updater(param);
        int round = 0;
        srand (time(NULL));
        LOG(INFO)<<"before train";
        float_type rmse = 0;
        {
            bool init_bin = 0;
            for (Tree &tree:trees) {
                {
                    TIMED_SCOPE(timerObj, "updateGH");
                    stats.updateGH(is_multi);
                }
                LOG(INFO)<<"after update GH";
                TIMED_SCOPE(timerObj, "construct tree");
                if(!init_bin) {
                    updater.use_similar_bundle = 0;
                    {
                        TIMED_SCOPE(timerObj, "init cut");
                        updater.init_cut(v_columns, stats, n_instances, columns);
                    }
                    LOG(INFO)<<"cut row"<<updater.v_cut[0].cut_row_ptr;
                    if(updater.use_similar_bundle)
                    {
                        TIMED_SCOPE(timerObj, "similar ins bundle");
                        updater.similar_ins_bundle(v_columns, stats, n_instances, dataSet, columns, iidold2new, is_multi);
                    }
                    init_bin = 1;
                }


                {
                    TIMED_SCOPE(timerObj, "grow");
                    updater.grow(tree, v_columns, stats);
                }
                {
                    TIMED_SCOPE(timerObj, "prune");
                    tree.prune_self(param.gamma);
                }

                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                predict_in_training(stats, tree);
                //next round
                round++;
                rmse = compute_rmse_bundle(stats, iidold2new, old_instances, dataSet);
                LOG(INFO) << "rmse = " << rmse;

            }
        }

        rmse = compute_rmse_bundle(stats, iidold2new, old_instances, dataSet);

        LOG(INFO) << "rmse = " << rmse;
        LOG(INFO)<<"stats_y:"<<stats.y;
        LOG(INFO)<<"stats_y_predict"<<stats.y_predict;
        delete []iidold2new;

        DataSet test_dataSet;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_hist(trees, test_dataSet, updater, y_predict);

        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;

        }

        float_type acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error = "<< (1 - acc);

        float_type sum_error = 0;
        for (int i = 0; i < test_instances; i++) {
            float_type e = y_predict[i] - test_dataSet.y()[i];
            sum_error += e * e;
        }
        float_type test_rmse = sqrt(sum_error / test_instances);
        LOG(INFO)<<"test rmse = "<<test_rmse;

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"train with hist:"<<(1-acc)<<std::endl;
        outfile.close();
        return rmse;
    }

    float_type train_hist_double(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);

        int n_instances = dataSet.n_instances();
        LOG(INFO)<<"before iidold2new";
        int old_instances = n_instances;
        int* iidold2new = new int [old_instances];
        LOG(INFO)<<"after iidold2new";
        thrust::sequence(iidold2new, iidold2new + old_instances);
        SyncArray<bool> is_multi(n_instances);

        InsStat stats;
        vector<Tree> trees;
        SparseColumns columns;
        columns.from_dataset(dataSet);
        trees.resize(param.n_trees);
        LOG(INFO)<<"before stats";
        stats.resize(n_instances);
        stats.y.copy_from(dataSet.y().data(), n_instances);
        LOG(INFO)<<"after stats";
        int n_devices = 1;

        vector<std::shared_ptr<SparseColumns>> v_columns;
        vector<std::shared_ptr<SparseColumns>> v_columns2;

        v_columns.resize(n_devices);
        v_columns2.resize(n_devices);
        for (int i = 0; i < n_devices; i++) {
            v_columns[i].reset(new SparseColumns());
            v_columns2[i].reset(new SparseColumns());
        }
        columns.to_multi_devices(v_columns);
        HistUpdater updater(param);
        int round = 0;
        srand (time(NULL));
        LOG(INFO)<<"before train";
        float_type rmse = 0;
        {
            bool init_bin = 0;
            for (Tree &tree:trees) {
                {
                    TIMED_SCOPE(timerObj, "updateGH");
                    stats.updateGH(is_multi);
                }
                LOG(INFO)<<"after update GH";
                TIMED_SCOPE(timerObj, "construct tree");
                if(!init_bin) {
                    updater.use_similar_bundle = 1;
                    {
                        TIMED_SCOPE(timerObj, "init cut");
                        updater.init_cut(v_columns, stats, n_instances, columns);
                    }
                    if(updater.use_similar_bundle)
                    {
                        TIMED_SCOPE(timerObj, "similar ins bundle");
                        updater.similar_ins_bundle(v_columns, v_columns2, stats, n_instances, dataSet, columns, iidold2new, is_multi);
                    }
                    init_bin = 1;
                }


                {
                    TIMED_SCOPE(timerObj, "grow");
                    if(round % 2 == 0)
                        updater.grow(tree, v_columns, stats);
                    else
                        updater.grow(tree, v_columns2, stats);
                }
                {
                    TIMED_SCOPE(timerObj, "prune");
                    tree.prune_self(param.gamma);
                }

                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                predict_in_training(stats, tree);
                //next round
                round++;
                rmse = compute_rmse_bundle(stats, iidold2new, old_instances, dataSet);
                LOG(INFO) << "rmse = " << rmse;

            }
        }

        rmse = compute_rmse_bundle(stats, iidold2new, old_instances, dataSet);

        LOG(INFO) << "rmse = " << rmse;
        LOG(INFO)<<"stats_y:"<<stats.y;
        LOG(INFO)<<"stats_y_predict"<<stats.y_predict;
        delete []iidold2new;


        DataSet test_dataSet;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_hist(trees, test_dataSet, updater, y_predict);

        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;

        }

        float_type acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error = "<<( 1 - acc);


        return rmse;
    }




    float_type train_hist_multi(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);


        int n_instances = dataSet.n_instances();
        LOG(INFO)<<"before iidold2new";
        int old_instances = n_instances;
        int* iidold2new = new int [old_instances];
        LOG(INFO)<<"after iidold2new";
        thrust::sequence(iidold2new, iidold2new + old_instances);
        SyncArray<bool> is_multi(n_instances);

        InsStat stats;
        vector<Tree> trees;
        SparseColumns columns;
        columns.from_dataset(dataSet);
        trees.resize(param.n_trees);
        LOG(INFO)<<"before stats";
        stats.resize(n_instances);
        stats.y.copy_from(dataSet.y().data(), n_instances);
        LOG(INFO)<<"after stats";
        int n_devices = 1;
        bool is_adopt = 0;
        bool is_front = 1;
        int numP = 2;

        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int i = 0; i < numP; i++){
            v_columns[i].resize(n_devices);
        }
        for(int pid = 0; pid < numP; pid++) {
            for (int i = 0; i < n_devices; i++) {
                v_columns[pid][i].reset(new SparseColumns());
            }
        }
        columns.to_multi_devices(v_columns[0]);
        HistUpdater updater(param);
        int round = 0;
        srand (time(NULL));
        LOG(INFO)<<"before train";
        float_type rmse = 0;

        {
            bool init_bin = 0;
            for (Tree &tree:trees) {
                {
                    TIMED_SCOPE(timerObj, "updateGH");
                    stats.updateGH(is_multi, numP);
                }
                LOG(INFO)<<"after update GH";
                TIMED_SCOPE(timerObj, "construct tree");
                if(!init_bin) {
                    updater.use_similar_bundle = 1;
                    {
                        TIMED_SCOPE(timerObj, "init cut");
                        updater.init_cut(v_columns[0], stats, n_instances, columns);
                    }
                    if(updater.use_similar_bundle)
                    {
                        TIMED_SCOPE(timerObj, "similar ins bundle");
                        bool is_random = 0;
                        updater.similar_ins_bundle_multi(v_columns, numP, stats, n_instances, dataSet, columns, iidold2new, is_multi, is_random);
                    }

                    init_bin = 1;
                }


                {
                    TIMED_SCOPE(timerObj, "grow");
                    if(is_adopt)
                        updater.grow(tree, v_columns[0], stats);
                    else {
                        if(!is_front)
                            updater.grow(tree, v_columns[round % numP], stats);
                        else {
                            updater.grow(tree, v_columns[round * numP / param.n_trees], stats);
                        }
                    }
                }
                {
                    TIMED_SCOPE(timerObj, "prune");
                    tree.prune_self(param.gamma);
                }

                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                predict_in_training(stats, tree);
                //next round
                round++;
                rmse = compute_rmse_bundle(stats, iidold2new, old_instances, dataSet);
                LOG(INFO) << "rmse = " << rmse;

            }
        }

        rmse = compute_rmse_bundle(stats, iidold2new, old_instances, dataSet);

        LOG(INFO) << "rmse = " << rmse;
        LOG(INFO)<<"stats_y:"<<stats.y;
        LOG(INFO)<<"stats_y_predict"<<stats.y_predict;
        delete []iidold2new;

        DataSet test_dataSet;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_hist(trees, test_dataSet, updater, y_predict);
        float_type sum_error = 0;
        for (int i = 0; i < test_instances; i++) {
            float_type e = y_predict[i] - test_dataSet.y()[i];
            sum_error += e * e;
        }
        float_type test_rmse = sqrt(sum_error / test_instances);

        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;

        }

        float_type acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error = "<<(1 - acc);
        LOG(INFO)<<"test rmse = "<<test_rmse;

        return rmse;
    }

    void split_to_multi_parts_evenly(vector<vector<std::shared_ptr<SparseColumns>>> &v_columns, int numP, InsStat &stats){
        SparseColumns& columns = *v_columns[0][0];
        float_type* csc_val_device = columns.csc_val.device_data();
        int* csc_row_device = columns.csc_row_ind.device_data();
        int* csc_col_device = columns.csc_col_ptr.device_data();
        return;
    }


    float_type train_hist_multi_independent(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);

        int n_instances = dataSet.n_instances();
        LOG(INFO)<<"before iidold2new";
        int old_instances = n_instances;
        int* iidold2new = new int [old_instances];
        LOG(INFO)<<"after iidold2new";
        thrust::sequence(iidold2new, iidold2new + old_instances);
        SyncArray<bool> is_multi(n_instances);

        InsStat stats;
        vector<Tree> trees;
        SparseColumns columns;
        columns.from_dataset(dataSet);
        trees.resize(param.n_trees);

        stats.resize(n_instances);
        stats.y.copy_from(dataSet.y().data(), n_instances);

        int n_devices = 1;
        bool is_adopt = 0;
        bool is_front = 1;
        bool is_independent = 1;
        int numP = 2;
        bool is_random = 0;
        bool weighted_gh = 1;

        vector<InsStat> stats_multi(numP);
        for(int i = 0; i < numP; i++){
            stats_multi[i].resize(n_instances);
            stats_multi[i].y.copy_from(dataSet.y().data(), n_instances);
        }

        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int i = 0; i < numP; i++){
            v_columns[i].resize(n_devices);
        }
        for(int pid = 0; pid < numP; pid++) {
            for (int i = 0; i < n_devices; i++) {
                v_columns[pid][i].reset(new SparseColumns());
            }
        }
        columns.to_multi_devices(v_columns[0]);
        HistUpdater updater(param);
        int round = 0;
        srand (time(NULL));
        LOG(INFO)<<"before train";
        float_type rmse = 0;

        {
            bool init_bin = 0;
            for (Tree &tree:trees) {
                {
                    TIMED_SCOPE(timerObj, "updateGH");
                    if(!is_front)
                        stats_multi[round % numP].updateGH(is_multi, numP);
                    else {
                        if(weighted_gh)
                            stats_multi[round * numP / param.n_trees].updateGH(is_multi, numP);
                        else
                            stats_multi[round * numP / param.n_trees].updateGH();
                    }
                }
                LOG(INFO)<<"after update GH";
                TIMED_SCOPE(timerObj, "construct tree");
                if(!init_bin) {
                    updater.use_similar_bundle = 1;
                    {
                        TIMED_SCOPE(timerObj, "init cut");
                        updater.init_cut(v_columns[0], stats_multi[0], n_instances, columns);
                    }
                    if(updater.use_similar_bundle)
                    {
                        TIMED_SCOPE(timerObj, "similar ins bundle");

                        updater.similar_ins_bundle_independent(v_columns, numP, stats_multi, n_instances, dataSet, columns, iidold2new, is_multi,is_random, weighted_gh);
                    }
                    init_bin = 1;
                }


                {
                    TIMED_SCOPE(timerObj, "grow");
                    if(is_adopt)
                        updater.grow(tree, v_columns[0], stats_multi[0]);
                    else{
                        if(!is_front)
                            updater.grow(tree, v_columns[round % numP], stats_multi[round % numP]);
                        else{
                            updater.grow(tree, v_columns[round * numP / param.n_trees], stats_multi[round * numP / param.n_trees]);
                        }
                    }

                }
                {
                    TIMED_SCOPE(timerObj, "prune");
                    tree.prune_self(param.gamma);
                }

                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                if(!is_front)
                    predict_in_training(stats_multi[round % numP], tree);
                else{
                    predict_in_training(stats_multi[round * numP / param.n_trees], tree);
                }
                //next round
                round++;
                LOG(INFO) << "rmse = " << rmse;

            }
        }

        LOG(INFO) << "rmse = " << rmse;
        delete []iidold2new;

        DataSet test_dataSet;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_hist(trees, test_dataSet, updater, y_predict, is_independent, numP);
        float_type sum_error = 0;
        for (int i = 0; i < test_instances; i++) {
            float_type e = y_predict[i] - test_dataSet.y()[i];
            sum_error += e * e;
        }
        float_type test_rmse = sqrt(sum_error / test_instances);

        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;

        }

        float_type acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error = "<<(1 - acc);
        LOG(INFO)<<"test rmse = "<<test_rmse;

        return rmse;
    }


    float_type compute_rmse(const InsStat &stats) {
        float_type sum_error = 0;
        const float_type *y_data = stats.y.host_data();
        const float_type *y_predict_data = stats.y_predict.host_data();
        for (int i = 0; i < stats.n_instances; ++i) {
            float_type e = y_predict_data[i] - y_data[i];
            sum_error += e * e;
        }
        float_type rmse = sqrt(sum_error / stats.n_instances);
        return rmse;
    }

    float_type compute_rmse_bundle(const InsStat &stats, int* iidold2new, int old_instances, DataSet& dataSet) {
        float_type sum_error = 0;
        const float_type *y_data = stats.y.host_data();
        const float_type *y_predict_data = stats.y_predict.host_data();
        for (int i = 0; i < old_instances; i++) {
            float_type e = y_predict_data[iidold2new[i]] - dataSet.y()[i];
            sum_error += e * e;
        }
        float_type rmse = sqrt(sum_error / old_instances);
        return rmse;
    }

    void predict_in_training(InsStat &stats, const Tree &tree) {

        TIMED_SCOPE(timerObj, "predict");
        float_type *y_predict_data = stats.y_predict.device_data();
        const int *nid_data = stats.nid.device_data();
        const Tree::TreeNode *nodes_data = tree.nodes.device_data();
        device_loop(stats.n_instances, [=]__device__(int i) {
            int nid = nid_data[i];
            while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
            y_predict_data[i] += nodes_data[nid].base_weight;
        });
    }

vector<float_type>& predict_test_dataset_rmse_hist(const vector<Tree> &trees, DataSet& dataSet, HistUpdater& updater,
        vector<float_type>& y_predict, bool is_independent = 0, int numP = 1) {
    size_t n_instances = dataSet.n_instances();

    SparseColumns columns;

    columns.from_dataset(dataSet);
    updater.get_bin_ids(columns);
    int n_column = columns.n_column;
    const int *iid_data = columns.csc_row_ind.host_data();
    const int *col_ptr_data = columns.csc_col_ptr.host_data();
    const int *bin_id_data = (*(updater.bin_id[0])).host_data();
    vector<int> nid_data(n_instances, 0);
    vector<int> nid_data_update(n_instances, 0);

    float_type rmse;
    for (int tid = 0; tid < trees.size() ; tid++) {
        const Tree &tree = trees[tid];
        const Tree::TreeNode *nodes_data = tree.nodes.host_data();
        for (int i = 0; i < n_instances; i++)
            nid_data[i] = 0;
        for (int depth = 0; depth < updater.depth; depth++) {
            int n_max_nodes_in_level = 1 << depth;//2^i
            int nid_offset = (1 << depth) - 1;//2^i - 1
            for (int iid = 0; iid < n_instances; iid++)
                nid_data_update[iid] = nid_data[iid];

//#pragma omp parallel for
            for (int col_id = 0; col_id < n_column; col_id++) {
                for (int fid = col_ptr_data[col_id]; fid < col_ptr_data[col_id + 1]; fid++) {


                    int iid = iid_data[fid];
                    int nid = nid_data[iid];

                    const Tree::TreeNode &node = nodes_data[nid];
                    if ((node.splittable()) && (node.split_feature_id == col_id)) {
                        if ((float_type) bin_id_data[fid] > node.split_value) {
                            //goes to left child
                            nid_data_update[iid] = node.lch_index;
                        } else {
                            //right child
                            nid_data_update[iid] = node.rch_index;
                        }
                    }
                }
            }

            for (int iid = 0; iid < n_instances; iid++) {
                nid_data[iid] = nid_data_update[iid];
            }
            for (int iid = 0; iid < n_instances; iid++) {
                int nid = nid_data[iid];
                if (nodes_data[nid].splittable() && (nid < nid_offset + n_max_nodes_in_level)) {
                    const Tree::TreeNode &node = nodes_data[nid];
                    if (node.default_right)
                        nid_data[iid] = node.rch_index;
                    else
                        nid_data[iid] = node.lch_index;
                }
            }
        }
        for (int iid = 0; iid < n_instances; iid++) {
            int nid = nid_data[iid];
            while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
            if(!is_independent)
                y_predict[iid] += nodes_data[nid].base_weight;
            else
                y_predict[iid] += nodes_data[nid].base_weight / numP;
        }
        float_type sum_error = 0;

        for (int i = 0; i < n_instances; i++) {
            float_type e = y_predict[i] - dataSet.y()[i];
            sum_error += e * e;
        }
        rmse = sqrt(sum_error / n_instances);
    }

    std::ofstream outfile;
    outfile.open(file_name, std::ios::out | std::ios::app);
    outfile<<"rmse: "<< rmse<<std::endl;
    outfile.close();
    return y_predict;
}

    float_type train_hist_only_one_party(GBMParam &param, int numP, int party_id) {
        cudaSetDevice(gpu_id);
        vector<DataSet> dataSets(numP);

        int num_bin = 64;

        for(int i = 0; i < numP; i++){
            dataSets[i].load_from_file(param.path+std::to_string(i + 1));
        }
        LOG(INFO)<<"after load datasets";
        int max_dimension = dataSets[0].n_features();

        //the number of instances of each dataset
        vector<int> n_instances(numP);
        vector<InsStat> stats(numP);
        int total_instances = 0;

        //the accumulate sum of number of instances
        vector<int> instance_accu(numP);
        instance_accu[0] = 0;
        for(int i = 0; i < numP; i++) {
            n_instances[i] = dataSets[i].n_instances();
            total_instances += n_instances[i];
            if(i!= 0) {
                instance_accu[i] += (n_instances[i-1] + instance_accu[i-1]);
            }
            stats[i].resize(n_instances[i]);
            stats[i].y.copy_from(dataSets[i].y().data(), n_instances[i]);
        }
        LOG(INFO)<<"after initial stats";
        //the stat of each company

        //the overall tree
        vector<Tree> trees;
        //csc of each dataset

        vector<SparseColumns> columns_csc(numP);
        vector<Csc2r> columns_csr(numP);

        for(int i = 0; i < numP; i++) {
            columns_csc[i].from_dataset(dataSets[i]);
            columns_csr[i].from_csr(columns_csc[i].csc_val.device_data(), columns_csc[i].csc_row_ind.device_data(),
                                    columns_csc[i].csc_col_ptr.device_data(), max_dimension, n_instances[i], columns_csc[i].nnz);
        }

        LOG(INFO)<<"after load columns";
        HistUpdater updater(param);
        //init the hash table
        updater.lsh_hash_init(500, 40, max_dimension, 1, 4.0, numP, lsh_seed);

        LOG(INFO)<<"after init hash";
        //the minimal value of each feature
        vector<float> min_fea(max_dimension);
        //the maximum value of each feature
        vector<float> max_fea(max_dimension);


        for(int fid = 0; fid < max_dimension; fid++) {
            min_fea[fid] = dataSets[0].min_fea[fid];
            max_fea[fid] = dataSets[0].max_fea[fid];
            for (int i = 1; i < numP; i++) {
                if(min_fea[fid] > dataSets[i].min_fea[fid]){
                    min_fea[fid] = dataSets[i].min_fea[fid];
                }
                if(max_fea[fid] < dataSets[i].max_fea[fid]){
                    max_fea[fid] = dataSets[i].max_fea[fid];
                }
            }
            if(min_fea[fid] == INFINITY || max_fea[fid] == -INFINITY){
                std::cout<<"error: empty dimension";
            }
        }
        LOG(INFO)<<"after init min max feature";

        //get bin id
        vector<vector<float>> bin_id_csr(numP);

        for(int i = 0; i < numP; i++){
            bin_id_csr[i].resize(columns_csr[i].csc_val.size());
            columns_csr[i].get_cut_points_evenly(num_bin, bin_id_csr[i], min_fea, max_fea);
        }


        LOG(INFO)<<"after get bin ids";

        //build the lsh table
        vector<SyncArray<int>> hash_values(numP);
        vector<SyncArray<float>> bin_id_csr_array(numP);
        for(int i = 0; i < numP; i++){
            bin_id_csr_array[i].resize(bin_id_csr[i].size());
            bin_id_csr_array[i].copy_from(bin_id_csr[i].data(), bin_id_csr[i].size());
        }
        //hash all datasets
        for(int i = 0; i < numP; i++){
            hash_values[i].resize(n_instances[i] * updater.lsh_table.param.n_table);
            updater.lsh_table.hash(n_instances[i], dataSets[i].n_features(), columns_csr[i].nnz, instance_accu[i],
                                   bin_id_csr_array[i], columns_csr[i].csc_col_ptr,
                                   columns_csr[i].csc_row_ind, hash_values[i], i);
            cudaDeviceSynchronize();
        }
        LOG(INFO)<<"hash values0:"<<hash_values[0];
        LOG(INFO)<<"hash values1:"<<hash_values[1];

        LOG(INFO)<<"after hash";
        //when train in company i, the similar id of other company's data in company i
        vector<vector<int>> similar_id(numP);
        vector<vector<int>> similar_nums(numP);
        for(int i = 0; i < numP; i++)
            similar_nums[i].resize(n_instances[i], 0);
        //get similar instance

        trees.resize(param.n_trees);

        //start id of trees of each company
        vector<int> n_tree(numP);
        n_tree[0] = 0;
        for(int i = 1; i < numP; i++){
            n_tree[i] = param.n_trees * n_instances[i - 1] / total_instances;
            n_tree[i] += n_tree[i - 1];
        }

        //sort the column
        int n_devices = 1;
        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int cid = 0; cid < numP; cid++){
            v_columns[cid].resize(n_devices);
            for (int i = 0; i < n_devices; i++)
                v_columns[cid][i].reset(new SparseColumns());
            columns_csc[cid].to_multi_devices(v_columns[cid]);
        }

        updater.v_cut.resize(n_devices);
        updater.v_cut[0].row_ptr.resize(max_dimension + 1);


        float* cut_val_host = updater.v_cut[0].cut_points.data();
        int* cut_row_host = updater.v_cut[0].row_ptr.data();
        cut_row_host[0] = 0;
        for(int fid = 0; fid < max_dimension; fid++){
            if(min_fea[fid] == INFINITY || max_fea[fid] == -INFINITY){
                cut_row_host[fid + 1] = cut_row_host[fid];
                std::cout<<"should not happen in dense"<<std::endl;
                continue;
            }
            updater.v_cut[0].cut_points.push_back(min_fea[fid] - (fabsf(min_fea[fid])+1e-5));

            for(int cid = 1; cid < num_bin; cid++){
                updater.v_cut[0].cut_points.push_back(min_fea[fid] + (max_fea[fid] - min_fea[fid]) / num_bin * cid);
            }
            cut_row_host[fid + 1] = cut_row_host[fid] + num_bin;
        }
        updater.v_cut[0].cut_row_ptr.resize(updater.v_cut[0].row_ptr.size());
        updater.v_cut[0].cut_row_ptr.copy_from(updater.v_cut[0].row_ptr.data(), updater.v_cut[0].row_ptr.size());
        updater.v_cut[0].cut_points_val.resize(updater.v_cut[0].cut_points.size());
        auto cut_points_val_ptr = updater.v_cut[0].cut_points_val.host_data();
        auto cut_row_ptr_data = updater.v_cut[0].cut_row_ptr.host_data();
        //descend order
        for(int i = 0; i < updater.v_cut[0].cut_row_ptr.size(); i++){
            int sum = cut_row_ptr_data[i] + cut_row_ptr_data[i+1] - 1;
            for(int j = cut_row_ptr_data[i+1] - 1; j >= cut_row_ptr_data[i]; j--)
                cut_points_val_ptr[j] = updater.v_cut[0].cut_points[sum - j];
        }
        LOG(INFO)<<"cut points val:"<<updater.v_cut[0].cut_points_val;
        LOG(INFO)<<"cut row ptr:"<<updater.v_cut[0].cut_row_ptr;




        //init bin id before the training
        vector<SyncArray<int>> bin_ids(numP);
        for(int i = 0; i < numP; i++){
            updater.init_bin_id_unsort(columns_csc[i], bin_ids[i]);

        }

        LOG(INFO)<<"bin ids 0:"<<bin_ids[0];
        int round = 0;
        //id of company training
        int train_cid = party_id;
        bool init_bin = 0;
        for(Tree &tree:trees){
            {
                TIMED_SCOPE(timerObj, "updateGH");
                stats[train_cid].updateGH();
            }

            LOG(INFO)<<"after update GH";
            if(!init_bin)
            {
                TIMED_SCOPE(timerObj, "init bin id");
                updater.init_bin_id(v_columns[train_cid]);
                init_bin = 1;
                LOG(INFO)<<"updater bin id:"<<(*updater.bin_id[0]);
            }
            {
                TIMED_SCOPE(timerObj, "grow");
                updater.grow(tree, v_columns[train_cid], stats[train_cid]);
            }
            {
                TIMED_SCOPE(timerObj, "prune");
                tree.prune_self(param.gamma);
            }

            LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
            predict_in_training(stats[train_cid], tree);
            for(int i = 0; i < numP; i++) {
                if(i != train_cid)
                    predict_the_company(tree, columns_csc[i], bin_ids[i], param.depth, stats[i], n_instances[i]);
            }
            //next round
            round++;
        }
        DataSet test_dataSet;
        test_dataSet.max_dimension = max_dimension;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_hist(trees, test_dataSet, updater, y_predict);
        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;
        }
        float acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error:"<<(1 - acc);

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"only party "<< party_id<<":"<<(1-acc)<<std::endl;
        outfile.close();
        return 0;
    }


    float_type train_hist_multi_party_new(GBMParam &param, int numP, float lsh_r, int lsh_n_table, int lsh_n_bucket, int max_dim) {
        cudaSetDevice(gpu_id);
        vector<DataSet> dataSets(numP);

        int num_bin = 256;

        for(int i = 0; i < numP; i++){
            dataSets[i].max_dimension = max_dim;
            dataSets[i].load_from_file(param.path+std::to_string(i));
        }
        LOG(INFO)<<"after load datasets";
        int max_dimension = max_dim;
        //the number of instances of each dataset
        vector<int> n_instances(numP);
        vector<InsStat> stats(numP);
        int total_instances = 0;

        //the accumulate sum of number of instances
        vector<int> instance_accu(numP);
        instance_accu[0] = 0;
        for(int i = 0; i < numP; i++) {
            n_instances[i] = dataSets[i].n_instances();
            total_instances += n_instances[i];
            if(i!= 0) {
                instance_accu[i] += (n_instances[i-1] + instance_accu[i-1]);
            }
            stats[i].resize(n_instances[i]);
            stats[i].y.copy_from(dataSets[i].y().data(), n_instances[i]);
        }
        LOG(INFO)<<"after initial stats";
        //the stat of each company

        //the overall tree
        vector<Tree> trees;
        //csc of each dataset

        vector<SparseColumns> columns_csc(numP);
//        vector<SparseColumns> columns_csr(numP);
        vector<Csc2r> columns_csr(numP);

        for(int i = 0; i < numP; i++) {
            columns_csc[i].from_dataset(dataSets[i]);
            columns_csr[i].from_csr(columns_csc[i].csc_val.device_data(), columns_csc[i].csc_row_ind.device_data(),
                                    columns_csc[i].csc_col_ptr.device_data(), max_dimension, n_instances[i], columns_csc[i].nnz);
        }

        LOG(INFO)<<"after load columns";
        HistUpdater updater(param);
        //init the hash table
        updater.lsh_hash_init(lsh_n_bucket, lsh_n_table, max_dimension, 1, lsh_r, numP, lsh_seed);
        LOG(INFO)<<"after init hash";
        //the minimal value of each feature
        vector<float> min_fea(max_dimension);
        //the maximum value of each feature
        vector<float> max_fea(max_dimension);


        for(int fid = 0; fid < max_dimension; fid++) {
            min_fea[fid] = dataSets[0].min_fea[fid];
            max_fea[fid] = dataSets[0].max_fea[fid];
            for (int i = 1; i < numP; i++) {
                if(min_fea[fid] > dataSets[i].min_fea[fid]){
                    min_fea[fid] = dataSets[i].min_fea[fid];
                }
                if(max_fea[fid] < dataSets[i].max_fea[fid]){
                    max_fea[fid] = dataSets[i].max_fea[fid];
                }
            }
            if(min_fea[fid] == INFINITY || max_fea[fid] == -INFINITY){
                std::cout<<"error: empty dimension";
            }
        }
        LOG(INFO)<<"after init min max feature";

        //get bin id
        //simply use the average number
        vector<vector<float>> bin_id_csr(numP);

        for(int i = 0; i < numP; i++){
            bin_id_csr[i].resize(columns_csr[i].csc_val.size());
            columns_csr[i].get_cut_points_evenly(num_bin, bin_id_csr[i], min_fea, max_fea);
        }


        LOG(INFO)<<"after get bin ids";


        //build the lsh table
        vector<SyncArray<int>> hash_values(numP);
        vector<SyncArray<float>> bin_id_csr_array(numP);
        for(int i = 0; i < numP; i++){
            bin_id_csr_array[i].resize(bin_id_csr[i].size());
            bin_id_csr_array[i].copy_from(bin_id_csr[i].data(), bin_id_csr[i].size());
        }
        //hash all datasets
        for(int i = 0; i < numP; i++){
            hash_values[i].resize(n_instances[i] * updater.lsh_table.param.n_table);
            updater.lsh_table.hash(n_instances[i], dataSets[i].n_features(), columns_csr[i].nnz, instance_accu[i],
                                   bin_id_csr_array[i], columns_csr[i].csc_col_ptr,
                                   columns_csr[i].csc_row_ind, hash_values[i], i);
            cudaDeviceSynchronize();
        }
        LOG(INFO)<<"hash values0:"<<hash_values[0];
        LOG(INFO)<<"hash values1:"<<hash_values[1];

        LOG(INFO)<<"after hash";
        vector<vector<int>> similar_id(numP);
        vector<vector<int>> similar_nums(numP);
        for(int i = 0; i < numP; i++)
            similar_nums[i].resize(n_instances[i], 0);
        for(int i = 0; i < numP; i++){
            float_type* stat_y_host_i = stats[i].y.host_data();
            for(int j = 0; j < numP; j++){
                if(j != i){
                    int* hash_value_host = hash_values[j].host_data();
                    vector<int> most_similar_id(n_instances[j]);
#pragma omp parallel for num_threads(20)
                    for(int iid = 0; iid < n_instances[j]; iid++){
                        vector<int> same_bucket_ins;
                        for(int tid = 0; tid < updater.lsh_table.param.n_table; tid++) {
                            int bid = hash_value_host[iid * updater.lsh_table.param.n_table + tid] % updater.lsh_table.param.n_bucket;
                            same_bucket_ins.insert(same_bucket_ins.end(),
                                                   updater.lsh_table.tables[tid][bid][i].begin(),
                                                   updater.lsh_table.tables[tid][bid][i].end());

                        }
                        //for a instance in company j, can not find similar instance in company i
                        if(same_bucket_ins.size() == 0){
                            ///to be solved
                            std::cout<<"error: no similar instance!!"<<std::endl;
                            continue;
                        }

                        std::sort(same_bucket_ins.begin(), same_bucket_ins.end());
                        float_type* stat_y_host_j = stats[j].y.host_data();
                        int max_count = 1;
                        int cur_count = 1;
                        int max_id = -1;

                        for(int sid = 1; sid < same_bucket_ins.size(); sid++){
                            if(same_bucket_ins[sid] == same_bucket_ins[sid-1])
                                cur_count++;
                            else {
                                if(cur_count > max_count){
                                    max_count = cur_count;
                                    max_id = same_bucket_ins[sid - 1];
                                }
                                cur_count = 1;
                            }
                        }
                        if(cur_count > max_count){
                            max_count = cur_count;
                            max_id = same_bucket_ins[same_bucket_ins.size() - 1];
                        }
                        most_similar_id[iid] = max_id - instance_accu[i];
                        similar_nums[i][max_id - instance_accu[i]]++;
                    }
                    similar_id[i].insert(similar_id[i].end(), most_similar_id.begin(), most_similar_id.end());
                }
            }
            CHECK(similar_id[i].size() == total_instances - n_instances[i]);
        }

        trees.resize(param.n_trees);

        //start id of trees of each company
        vector<int> n_tree(numP);
        n_tree[0] = 0;
        for(int i = 1; i < numP; i++){
            n_tree[i] = param.n_trees * n_instances[i - 1] / total_instances;
            n_tree[i] += n_tree[i - 1];
        }

        //sort the column
        int n_devices = 1;
        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int cid = 0; cid < numP; cid++){
            v_columns[cid].resize(n_devices);
            for (int i = 0; i < n_devices; i++)
                v_columns[cid][i].reset(new SparseColumns());
            columns_csc[cid].to_multi_devices(v_columns[cid]);
        }


        updater.v_cut.resize(n_devices);
        updater.v_cut[0].row_ptr.resize(max_dimension + 1);


        float* cut_val_host = updater.v_cut[0].cut_points.data();
        int* cut_row_host = updater.v_cut[0].row_ptr.data();
        cut_row_host[0] = 0;
        for(int fid = 0; fid < max_dimension; fid++){
            if(min_fea[fid] == INFINITY || max_fea[fid] == -INFINITY){
                cut_row_host[fid + 1] = cut_row_host[fid];
                std::cout<<"should not happen in dense"<<std::endl;
                continue;
            }
            updater.v_cut[0].cut_points.push_back(min_fea[fid] - (fabsf(min_fea[fid])+1e-5));

            for(int cid = 1; cid < num_bin; cid++){
                updater.v_cut[0].cut_points.push_back(min_fea[fid] + (max_fea[fid] - min_fea[fid]) / num_bin * cid);
            }
            cut_row_host[fid + 1] = cut_row_host[fid] + num_bin;
        }
        updater.v_cut[0].cut_row_ptr.resize(updater.v_cut[0].row_ptr.size());
        updater.v_cut[0].cut_row_ptr.copy_from(updater.v_cut[0].row_ptr.data(), updater.v_cut[0].row_ptr.size());
        updater.v_cut[0].cut_points_val.resize(updater.v_cut[0].cut_points.size());
        auto cut_points_val_ptr = updater.v_cut[0].cut_points_val.host_data();
        auto cut_row_ptr_data = updater.v_cut[0].cut_row_ptr.host_data();
        //descend order
        for(int i = 0; i < updater.v_cut[0].cut_row_ptr.size(); i++){
            int sum = cut_row_ptr_data[i] + cut_row_ptr_data[i+1] - 1;
            for(int j = cut_row_ptr_data[i+1] - 1; j >= cut_row_ptr_data[i]; j--)
                cut_points_val_ptr[j] = updater.v_cut[0].cut_points[sum - j];
        }
        LOG(INFO)<<"cut points val:"<<updater.v_cut[0].cut_points_val;
        LOG(INFO)<<"cut row ptr:"<<updater.v_cut[0].cut_row_ptr;

        //init bin id before the training
        vector<SyncArray<int>> bin_ids(numP);
        for(int i = 0; i < numP; i++){
            updater.init_bin_id_unsort(columns_csc[i], bin_ids[i]);

        }

        LOG(INFO)<<"bin ids 0:"<<bin_ids[0];
        int round = 0;
        //id of company training
        int train_cid = 0;
        bool init_bin = 0;
        for(Tree &tree:trees){
            {
                TIMED_SCOPE(timerObj, "updateGH");
                updateGH_eachtree(round, numP, init_bin, stats, similar_id, train_cid, n_tree, n_instances);
            }

            LOG(INFO)<<"after update GH";
            if(!init_bin)
            {
                TIMED_SCOPE(timerObj, "init bin id");
                updater.init_bin_id(v_columns[train_cid]);
                init_bin = 1;
                LOG(INFO)<<"updater bin id:"<<(*updater.bin_id[0]);
            }
            {
                TIMED_SCOPE(timerObj, "grow");
                updater.grow(tree, v_columns[train_cid], stats[train_cid]);
            }
            {
                TIMED_SCOPE(timerObj, "prune");
                tree.prune_self(param.gamma);
            }

            LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
            predict_in_training(stats[train_cid], tree);
            for(int i = 0; i < numP; i++) {
                if(i != train_cid)
                    predict_the_company(tree, columns_csc[i], bin_ids[i], param.depth, stats[i], n_instances[i]);
            }
            //next round
            round++;
        }
        DataSet test_dataSet;
        test_dataSet.max_dimension = max_dimension;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_hist(trees, test_dataSet, updater, y_predict);
        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;
        }
        float acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error:"<<(1 - acc);
        std::cout<<"test error:"<<(1-acc)<<std::endl;
        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"train with "<<numP<<" party:"<<(1-acc)<<std::endl;
        outfile.close();
        return 0;
    }

    float_type train_exact_multi_party(GBMParam &param, int numP,float lsh_r, int lsh_n_table, int lsh_n_bucket) {
        TIMED_SCOPE(timerObj,"total_training_time");
        cudaSetDevice(gpu_id);
        vector<DataSet> dataSets(numP);

        for(int i = 0; i < numP; i++){
            std::cout<<"i:"<<std::to_string(i);
            dataSets[i].load_from_file(param.path+std::to_string(i));
        }
        LOG(INFO)<<"after load datasets";
        //to be revised
        int max_dimension = dataSets[0].n_features();

        //the number of instances of each dataset
        vector<int> n_instances(numP);
        vector<InsStat> stats(numP);
        int total_instances = 0;

        //the accumulate sum of number of instances
        vector<int> instance_accu(numP);
        instance_accu[0] = 0;
        for(int i = 0; i < numP; i++) {
            n_instances[i] = dataSets[i].n_instances();
            total_instances += n_instances[i];
            if(i!= 0) {
                instance_accu[i] += (n_instances[i-1] + instance_accu[i-1]);
            }
            stats[i].resize(n_instances[i]);
            stats[i].y.copy_from(dataSets[i].y().data(), n_instances[i]);
        }
        LOG(INFO)<<"after initial stats";
        //the stat of each company

        //the overall tree
        vector<Tree> trees;
        //csc of each dataset

        vector<SparseColumns> columns_csc(numP);
        vector<Csc2r> columns_csr(numP);

        for(int i = 0; i < numP; i++) {
            columns_csc[i].from_dataset(dataSets[i]);
            columns_csr[i].from_csr(columns_csc[i].csc_val.device_data(), columns_csc[i].csc_row_ind.device_data(),
                                    columns_csc[i].csc_col_ptr.device_data(), max_dimension, n_instances[i], columns_csc[i].nnz);
        }

        LOG(INFO)<<"after load columns";
        ExactUpdater updater(param);
        //init the hash table
        updater.lsh_hash_init(lsh_n_bucket, lsh_n_table, max_dimension, 1, lsh_r, numP, lsh_seed);

        LOG(INFO)<<"after init hash";


        //build the lsh table
        vector<SyncArray<int>> hash_values(numP);

        //hash all datasets
        for(int i = 0; i < numP; i++){
            hash_values[i].resize(n_instances[i] * updater.lsh_table.param.n_table);
            updater.lsh_table.hash(n_instances[i], dataSets[i].n_features(), columns_csr[i].nnz, instance_accu[i],
                                   columns_csr[i].csc_val, columns_csr[i].csc_col_ptr,
                                   columns_csr[i].csc_row_ind, hash_values[i], i);
            cudaDeviceSynchronize();
        }

        LOG(INFO)<<"after hash";
        //when train in company i, the similar id of other company's data in company i
        vector<vector<int>> similar_id(numP);
        vector<vector<int>> similar_nums(numP);
        for(int i = 0; i < numP; i++)
            similar_nums[i].resize(n_instances[i], 0);
        //get similar instance

        {TIMED_SCOPE(timerObj,"get similar ids");
#pragma omp parallel for
        for(int i = 0; i < numP; i++){
            for(int j = 0; j < numP; j++){
                if(j != i){
                    TIMED_SCOPE(timerBlkObj, "begin in a party loop");
                    int* hash_value_host = hash_values[j].host_data();
                    vector<int> most_similar_id(n_instances[j]);
                    for(int iid = 0; iid < n_instances[j]; iid++){
                        PERFORMANCE_CHECKPOINT_WITH_ID(timerBlkObj, "begin in an instance loop");
                        vector<int> same_bucket_ins;
                        for(int tid = 0; tid < updater.lsh_table.param.n_table; tid++) {
                            int bid = hash_value_host[iid * updater.lsh_table.param.n_table + tid];
                            same_bucket_ins.insert(same_bucket_ins.end(),
                                                   updater.lsh_table.tables[tid][bid][i].begin(),
                                                   updater.lsh_table.tables[tid][bid][i].end());
                        }

                        //for a instance in company j, can not find similar instance in company i


                        if(same_bucket_ins.size() == 0){
                            ///to be solved
                            std::cout<<"error: no similar instance!!"<<std::endl;
                            continue;
                        }

                        PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "get same bucket ins");


                        //gpu version

                        SyncArray<int> same_bucket_ins_array(same_bucket_ins.size());
                        same_bucket_ins_array.copy_from(same_bucket_ins.data(), same_bucket_ins.size());
                        thrust::sort(thrust::cuda::par, same_bucket_ins_array.device_data(),
                                     same_bucket_ins_array.device_end());

                        PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "sort same bucket ins");


                        SyncArray<int> count_fre(same_bucket_ins.size());
                        thrust::fill(thrust::cuda::par, count_fre.device_data(), count_fre.device_end(), 1);

                        auto new_end = thrust::reduce_by_key(thrust::cuda::par, same_bucket_ins_array.device_data(), same_bucket_ins_array.device_end(),
                                              count_fre.device_data(), same_bucket_ins_array.device_data(), count_fre.device_data());

                        int position = thrust::max_element(thrust::cuda::par, count_fre.device_data(), new_end.second) - count_fre.device_data();

                        int max_id_cpu[1];
                        cudaMemcpy(max_id_cpu, same_bucket_ins_array.device_data() + position, sizeof(int), cudaMemcpyDeviceToHost);
                        int max_id = max_id_cpu[0];

                        PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "get max id");
                        most_similar_id[iid] = max_id - instance_accu[i];



                    }



                    similar_id[i].insert(similar_id[i].end(), most_similar_id.begin(), most_similar_id.end());


                }
            }
            CHECK(similar_id[i].size() == total_instances - n_instances[i]);
        }
        }
        std::cout<<"after get similar"<<std::endl;
        trees.resize(param.n_trees);

        //start id of trees of each company
        vector<int> n_tree(numP);
        n_tree[0] = 0;
        for(int i = 1; i < numP; i++){
            n_tree[i] = param.n_trees * n_instances[i - 1] / total_instances;
            n_tree[i] += n_tree[i - 1];
        }

        //sort the column
        int n_devices = 1;
        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int cid = 0; cid < numP; cid++){
            v_columns[cid].resize(n_devices);
            for (int i = 0; i < n_devices; i++)
                v_columns[cid][i].reset(new SparseColumns());
            columns_csc[cid].to_multi_devices(v_columns[cid]);
        }

        int round = 0;
        //id of company training
        int train_cid = 0;

        DataSet test_dataSet;
        test_dataSet.max_dimension = max_dimension;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);

        for(Tree &tree:trees){
            LOG(INFO)<<"round:"<<round;
            TIMED_SCOPE(timerObj, "EachTree");
            {
                TIMED_SCOPE(timerObj, "updateGH");
                updateGH_eachtree_exact(round, numP, stats, similar_id, train_cid, n_tree, n_instances);
            }

            LOG(INFO)<<"after update GH";
            {
                TIMED_SCOPE(timerObj, "grow");
                updater.grow(tree, v_columns[train_cid], stats[train_cid]);
            }
            {
                TIMED_SCOPE(timerObj, "prune");
                tree.prune_self(param.gamma);
            }
            tree.shrink(param.learning_rate);
            LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
            predict_in_training(stats[train_cid], tree);
            for(int i = 0; i < numP; i++) {
                if(i != train_cid)
                    predict_the_company_exact(tree, columns_csc[i], param.depth, stats[i], n_instances[i]);
            }
            predict_test_dataset_rmse_exact(trees, test_dataSet, updater, y_predict, round+1);
            int right_num = 0;
            for(int i = 0; i < test_instances; i++){
                if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                    right_num++;
            }
            float acc = right_num * 1.0 / test_instances;
            LOG(INFO)<<"test error:"<<(1 - acc);
            //next round
            round++;
        }

        predict_test_dataset_rmse_exact(trees, test_dataSet, updater, y_predict);
        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;
        }
        float acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"final test error:"<<(1 - acc);

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"train with "<<numP<<" party:"<<(1-acc)<<std::endl;
        outfile.close();

        return 0;
    }


    float_type train_exact_multi_party_normalized(GBMParam &param, int numP) {
        cudaSetDevice(gpu_id);
        vector<DataSet> dataSets(numP);


        int num_bin = 64;

        for(int i = 0; i < numP; i++){
            dataSets[i].load_from_file(param.path+std::to_string(i + 1));
        }
        LOG(INFO)<<"after load datasets";
        //to be revised
        int max_dimension = dataSets[0].n_features();

        //the number of instances of each dataset
        vector<int> n_instances(numP);
        vector<InsStat> stats(numP);
        int total_instances = 0;

        //the accumulate sum of number of instances
        vector<int> instance_accu(numP);
        instance_accu[0] = 0;
        for(int i = 0; i < numP; i++) {
            n_instances[i] = dataSets[i].n_instances();
            total_instances += n_instances[i];
            if(i!= 0) {
                instance_accu[i] += (n_instances[i-1] + instance_accu[i-1]);
            }
            stats[i].resize(n_instances[i]);
            stats[i].y.copy_from(dataSets[i].y().data(), n_instances[i]);
//            stats[i].updateGH();
        }
        LOG(INFO)<<"after initial stats";
        //the stat of each company

        //the overall tree
        vector<Tree> trees;
        //csc of each dataset

        vector<SparseColumns> columns_csc(numP);
        vector<Csc2r> columns_csr(numP);

        for(int i = 0; i < numP; i++) {
            columns_csc[i].from_dataset(dataSets[i]);
            columns_csr[i].from_csr(columns_csc[i].csc_val.device_data(), columns_csc[i].csc_row_ind.device_data(),
                                    columns_csc[i].csc_col_ptr.device_data(), max_dimension, n_instances[i], columns_csc[i].nnz);
        }

        LOG(INFO)<<"after load columns";
        ExactUpdater updater(param);
        //init the hash table
        updater.lsh_hash_init(500, 40, max_dimension, 1, 4.0, numP, lsh_seed);

        LOG(INFO)<<"after init hash";

        //the minimal value of each feature
        vector<float> min_fea(max_dimension);
        //the maximum value of each feature
        vector<float> max_fea(max_dimension);


        for(int fid = 0; fid < max_dimension; fid++) {
            min_fea[fid] = dataSets[0].min_fea[fid];
            max_fea[fid] = dataSets[0].max_fea[fid];
            for (int i = 1; i < numP; i++) {
                if(min_fea[fid] > dataSets[i].min_fea[fid]){
                    min_fea[fid] = dataSets[i].min_fea[fid];
                }
                if(max_fea[fid] < dataSets[i].max_fea[fid]){
                    max_fea[fid] = dataSets[i].max_fea[fid];
                }
            }
            if(min_fea[fid] == INFINITY || max_fea[fid] == -INFINITY){
                std::cout<<"error: empty dimension";
            }
        }
        LOG(INFO)<<"after init min max feature";

        //get bin id
        //simply use the average number
        vector<vector<float>> bin_id_csr(numP);

        for(int i = 0; i < numP; i++){
            bin_id_csr[i].resize(columns_csr[i].csc_val.size());
            columns_csr[i].get_cut_points_evenly(num_bin, bin_id_csr[i], min_fea, max_fea);
        }


        LOG(INFO)<<"after get bin ids";


        //build the lsh table
        vector<SyncArray<int>> hash_values(numP);
        vector<SyncArray<float>> bin_id_csr_array(numP);
        for(int i = 0; i < numP; i++){
            bin_id_csr_array[i].resize(bin_id_csr[i].size());
            bin_id_csr_array[i].copy_from(bin_id_csr[i].data(), bin_id_csr[i].size());
        }


        //hash all datasets
        for(int i = 0; i < numP; i++){
            hash_values[i].resize(n_instances[i] * updater.lsh_table.param.n_table);
            updater.lsh_table.hash(n_instances[i], dataSets[i].n_features(), columns_csr[i].nnz, instance_accu[i],
                                   bin_id_csr_array[i], columns_csr[i].csc_col_ptr,
                                   columns_csr[i].csc_row_ind, hash_values[i], i);
            cudaDeviceSynchronize();
        }
        LOG(INFO)<<"hash values0:"<<hash_values[0];
        LOG(INFO)<<"hash values1:"<<hash_values[1];

        LOG(INFO)<<"after hash";
        //when train in company i, the similar id of other company's data in company i
        vector<vector<int>> similar_id(numP);
        vector<vector<int>> similar_nums(numP);
        for(int i = 0; i < numP; i++)
            similar_nums[i].resize(n_instances[i], 0);
        //get similar instance
        for(int i = 0; i < numP; i++){
            float_type* stat_y_host_i = stats[i].y.host_data();
            for(int j = 0; j < numP; j++){
                if(j != i){
                    int* hash_value_host = hash_values[j].host_data();
                    vector<int> most_similar_id(n_instances[j]);
#pragma omp parallel for num_threads(20)
                    for(int iid = 0; iid < n_instances[j]; iid++){
                        vector<int> same_bucket_ins;
                        for(int tid = 0; tid < updater.lsh_table.param.n_table; tid++) {
                            int bid = hash_value_host[iid * updater.lsh_table.param.n_table + tid] % updater.lsh_table.param.n_bucket;
                            same_bucket_ins.insert(same_bucket_ins.end(),
                                                   updater.lsh_table.tables[tid][bid][i].begin(),
                                                   updater.lsh_table.tables[tid][bid][i].end());

                        }
                        //for a instance in company j, can not find similar instance in company i
                        if(same_bucket_ins.size() == 0){
                            ///to be solved
                            std::cout<<"error: no similar instance!!"<<std::endl;
                            continue;
                        }

                        std::sort(same_bucket_ins.begin(), same_bucket_ins.end());
                        float_type* stat_y_host_j = stats[j].y.host_data();
                        int max_count = 1;
                        int cur_count = 1;
                        int max_id = -1;

                        for(int sid = 1; sid < same_bucket_ins.size(); sid++){
                            ///can add: whether the label are the same
                            if(same_bucket_ins[sid] == same_bucket_ins[sid-1])
                                cur_count++;
                            else {
                                if(cur_count > max_count){
                                    max_count = cur_count;
                                    max_id = same_bucket_ins[sid - 1];
                                }
                                cur_count = 1;
                            }
                        }
                        if(cur_count > max_count){
                            max_count = cur_count;
                            max_id = same_bucket_ins[same_bucket_ins.size() - 1];
                        }
                        most_similar_id[iid] = max_id - instance_accu[i];
                        similar_nums[i][max_id - instance_accu[i]]++;
                    }
                    similar_id[i].insert(similar_id[i].end(), most_similar_id.begin(), most_similar_id.end());
                }
            }
            CHECK(similar_id[i].size() == total_instances - n_instances[i]);
        }

        trees.resize(param.n_trees);

        //start id of trees of each company
        vector<int> n_tree(numP);
        n_tree[0] = 0;
        for(int i = 1; i < numP; i++){
            n_tree[i] = param.n_trees * n_instances[i - 1] / total_instances;
            n_tree[i] += n_tree[i - 1];
        }

        //sort the column
        int n_devices = 1;
        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int cid = 0; cid < numP; cid++){
            v_columns[cid].resize(n_devices);
            for (int i = 0; i < n_devices; i++)
                v_columns[cid][i].reset(new SparseColumns());
            columns_csc[cid].to_multi_devices(v_columns[cid]);
        }

        int round = 0;
        //id of company training
        int train_cid = 0;

        for(Tree &tree:trees){
            {
                TIMED_SCOPE(timerObj, "updateGH");
                updateGH_eachtree_exact(round, numP, stats, similar_id, train_cid, n_tree, n_instances);
            }

            LOG(INFO)<<"after update GH";
            {
                TIMED_SCOPE(timerObj, "grow");
                updater.grow(tree, v_columns[train_cid], stats[train_cid]);
            }
            {
                TIMED_SCOPE(timerObj, "prune");
                tree.prune_self(param.gamma);
            }

            LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
            predict_in_training(stats[train_cid], tree);
            for(int i = 0; i < numP; i++) {
                if(i != train_cid)
                    predict_the_company_exact(tree, columns_csc[i], param.depth, stats[i], n_instances[i]);
            }
            //next round
            round++;
        }
        DataSet test_dataSet;
        test_dataSet.max_dimension = max_dimension;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_exact(trees, test_dataSet, updater, y_predict);
        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;
        }
        float acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error:"<<(1 - acc);

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"train with "<<numP<<" party:"<<(1-acc)<<std::endl;
        outfile.close();
        return 0;
    }

    float_type train_exact_multi_party_naive(GBMParam &param, int numP) {
        cudaSetDevice(gpu_id);
        vector<DataSet> dataSets(numP);

        for(int i = 0; i < numP; i++){
            dataSets[i].load_from_file(param.path+std::to_string(i % 10));
        }

        LOG(INFO)<<"after load datasets";
        int max_dimension = dataSets[0].n_features();

        //the number of instances of each dataset
        vector<int> n_instances(numP);
        vector<InsStat> stats(numP);
        int total_instances = 0;

        //the accumulate sum of number of instances
        vector<int> instance_accu(numP);
        instance_accu[0] = 0;
        for(int i = 0; i < numP; i++) {
            n_instances[i] = dataSets[i].n_instances();
            total_instances += n_instances[i];
            if(i!= 0) {
                instance_accu[i] += (n_instances[i-1] + instance_accu[i-1]);
            }
            stats[i].resize(n_instances[i]);
            stats[i].y.copy_from(dataSets[i].y().data(), n_instances[i]);
        }
        LOG(INFO)<<"after initial stats";
        //the stat of each company

        //the overall tree
        vector<Tree> trees;
        //csc of each dataset

        vector<SparseColumns> columns_csc(numP);
        vector<Csc2r> columns_csr(numP);

        for(int i = 0; i < numP; i++) {
            columns_csc[i].from_dataset(dataSets[i]);
            columns_csr[i].from_csr(columns_csc[i].csc_val.device_data(), columns_csc[i].csc_row_ind.device_data(),
                                    columns_csc[i].csc_col_ptr.device_data(), max_dimension, n_instances[i], columns_csc[i].nnz);
        }

        LOG(INFO)<<"after load columns";
        ExactUpdater updater(param);

        trees.resize(param.n_trees);

        //start id of trees of each company
        vector<int> n_tree(numP);
        n_tree[0] = 0;
        for(int i = 1; i < numP; i++){
            n_tree[i] = param.n_trees * n_instances[i - 1] / total_instances;
            n_tree[i] += n_tree[i - 1];
        }

        //sort the column
        int n_devices = 1;
        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int cid = 0; cid < numP; cid++){
            v_columns[cid].resize(n_devices);
            for (int i = 0; i < n_devices; i++)
                v_columns[cid][i].reset(new SparseColumns());
            columns_csc[cid].to_multi_devices(v_columns[cid]);
        }

        int round = 0;
        //id of company training
        int train_cid = 0;
        DataSet test_dataSet;
        test_dataSet.max_dimension = max_dimension;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        for(Tree &tree:trees){
            {
                TIMED_SCOPE(timerObj, "updateGH");
                updateGH_eachtree_naive(round, numP, stats, train_cid, n_tree, n_instances);
            }

            LOG(INFO)<<"after update GH";
            {
                TIMED_SCOPE(timerObj, "grow");
                updater.grow(tree, v_columns[train_cid], stats[train_cid]);
            }
            {
                TIMED_SCOPE(timerObj, "prune");
                tree.prune_self(param.gamma);
            }
            tree.shrink(param.learning_rate);
            LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
            predict_in_training(stats[train_cid], tree);
            for(int i = 0; i < numP; i++) {
                if(i != train_cid)
                    predict_the_company_exact(tree, columns_csc[i], param.depth, stats[i], n_instances[i]);
            }
            predict_test_dataset_rmse_exact(trees, test_dataSet, updater, y_predict, round+1);
            int right_num = 0;
            for(int i = 0; i < test_instances; i++){
                if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                    right_num++;
            }
            float acc = right_num * 1.0 / test_instances;
            LOG(INFO)<<"test error:"<<(1 - acc);

            round++;
        }

        predict_test_dataset_rmse_exact(trees, test_dataSet, updater, y_predict);
        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;
        }
        float acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error:"<<(1 - acc);

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"train with "<<numP<<" party:"<<(1-acc)<<std::endl;
        outfile.close();
        return 0;
    }


    float_type train_hist_multi_party_naive(GBMParam &param, int numP) {
        cudaSetDevice(gpu_id);
        vector<DataSet> dataSets(numP);



        int num_bin = 64;


        for(int i = 0; i < numP; i++){
            dataSets[i].load_from_file(param.path+std::to_string(i/10) + std::to_string(i % 10));
        }

        LOG(INFO)<<"after load datasets";
        int max_dimension = dataSets[0].n_features();

        //the number of instances of each dataset
        vector<int> n_instances(numP);
        vector<InsStat> stats(numP);
        int total_instances = 0;

        vector<int> instance_accu(numP);
        instance_accu[0] = 0;
        for(int i = 0; i < numP; i++) {
            n_instances[i] = dataSets[i].n_instances();
            total_instances += n_instances[i];
            if(i!= 0) {
                instance_accu[i] += (n_instances[i-1] + instance_accu[i-1]);
            }
            stats[i].resize(n_instances[i]);
            stats[i].y.copy_from(dataSets[i].y().data(), n_instances[i]);
        }
        LOG(INFO)<<"after initial stats";

        //the overall tree
        vector<Tree> trees;
        //csc of each dataset

        vector<SparseColumns> columns_csc(numP);
        vector<Csc2r> columns_csr(numP);

        for(int i = 0; i < numP; i++) {
            columns_csc[i].from_dataset(dataSets[i]);
            columns_csr[i].from_csr(columns_csc[i].csc_val.device_data(), columns_csc[i].csc_row_ind.device_data(),
                                    columns_csc[i].csc_col_ptr.device_data(), max_dimension, n_instances[i], columns_csc[i].nnz);
        }

        LOG(INFO)<<"after load columns";
        HistUpdater updater(param);
        //init the hash table
        updater.lsh_hash_init(500, 40, max_dimension, 1, 4.0, numP, lsh_seed);

        LOG(INFO)<<"after init hash";
        //the minimal value of each feature
        vector<float> min_fea(max_dimension);
        //the maximum value of each feature
        vector<float> max_fea(max_dimension);


        for(int fid = 0; fid < max_dimension; fid++) {
            min_fea[fid] = dataSets[0].min_fea[fid];
            max_fea[fid] = dataSets[0].max_fea[fid];
            for (int i = 1; i < numP; i++) {
                if(min_fea[fid] > dataSets[i].min_fea[fid]){
                    min_fea[fid] = dataSets[i].min_fea[fid];
                }
                if(max_fea[fid] < dataSets[i].max_fea[fid]){
                    max_fea[fid] = dataSets[i].max_fea[fid];
                }
            }
            if(min_fea[fid] == INFINITY || max_fea[fid] == -INFINITY){
                std::cout<<"error: empty dimension";
            }
        }
        LOG(INFO)<<"after init min max feature";

        //get bin id
        vector<vector<float>> bin_id_csr(numP);

        for(int i = 0; i < numP; i++){
            bin_id_csr[i].resize(columns_csr[i].csc_val.size());
            columns_csr[i].get_cut_points_evenly(num_bin, bin_id_csr[i], min_fea, max_fea);
        }


        LOG(INFO)<<"after get bin ids";


        //build the lsh table
        vector<SyncArray<int>> hash_values(numP);
        vector<SyncArray<float>> bin_id_csr_array(numP);
        for(int i = 0; i < numP; i++){
            bin_id_csr_array[i].resize(bin_id_csr[i].size());
            bin_id_csr_array[i].copy_from(bin_id_csr[i].data(), bin_id_csr[i].size());
        }
        //hash all datasets
        for(int i = 0; i < numP; i++){
            hash_values[i].resize(n_instances[i] * updater.lsh_table.param.n_table);
            updater.lsh_table.hash(n_instances[i], dataSets[i].n_features(), columns_csr[i].nnz, instance_accu[i],
                                   bin_id_csr_array[i], columns_csr[i].csc_col_ptr,
                                   columns_csr[i].csc_row_ind, hash_values[i], i);
            cudaDeviceSynchronize();
        }
        LOG(INFO)<<"hash values0:"<<hash_values[0];
        LOG(INFO)<<"hash values1:"<<hash_values[1];

        LOG(INFO)<<"after hash";
        
        trees.resize(param.n_trees);

        //start id of trees of each company
        vector<int> n_tree(numP);
        n_tree[0] = 0;
        for(int i = 1; i < numP; i++){
            n_tree[i] = param.n_trees * n_instances[i - 1] / total_instances;
            n_tree[i] += n_tree[i - 1];
        }

        //sort the column
        int n_devices = 1;
        vector<vector<std::shared_ptr<SparseColumns>>> v_columns(numP);
        for(int cid = 0; cid < numP; cid++){
            v_columns[cid].resize(n_devices);
            for (int i = 0; i < n_devices; i++)
                v_columns[cid][i].reset(new SparseColumns());
            columns_csc[cid].to_multi_devices(v_columns[cid]);
        }


        updater.v_cut.resize(n_devices);
        updater.v_cut[0].row_ptr.resize(max_dimension + 1);


        float* cut_val_host = updater.v_cut[0].cut_points.data();
        int* cut_row_host = updater.v_cut[0].row_ptr.data();
        cut_row_host[0] = 0;
        for(int fid = 0; fid < max_dimension; fid++){
            if(min_fea[fid] == INFINITY || max_fea[fid] == -INFINITY){
                cut_row_host[fid + 1] = cut_row_host[fid];
                std::cout<<"should not happen in dense"<<std::endl;
                continue;
            }
            updater.v_cut[0].cut_points.push_back(min_fea[fid] - (fabsf(min_fea[fid])+1e-5));

            for(int cid = 1; cid < num_bin; cid++){
                updater.v_cut[0].cut_points.push_back(min_fea[fid] + (max_fea[fid] - min_fea[fid]) / num_bin * cid);
            }
            cut_row_host[fid + 1] = cut_row_host[fid] + num_bin;
        }
        updater.v_cut[0].cut_row_ptr.resize(updater.v_cut[0].row_ptr.size());
        updater.v_cut[0].cut_row_ptr.copy_from(updater.v_cut[0].row_ptr.data(), updater.v_cut[0].row_ptr.size());
        updater.v_cut[0].cut_points_val.resize(updater.v_cut[0].cut_points.size());
        auto cut_points_val_ptr = updater.v_cut[0].cut_points_val.host_data();
        auto cut_row_ptr_data = updater.v_cut[0].cut_row_ptr.host_data();
        //descend order
        for(int i = 0; i < updater.v_cut[0].cut_row_ptr.size(); i++){
            int sum = cut_row_ptr_data[i] + cut_row_ptr_data[i+1] - 1;
            for(int j = cut_row_ptr_data[i+1] - 1; j >= cut_row_ptr_data[i]; j--)
                cut_points_val_ptr[j] = updater.v_cut[0].cut_points[sum - j];
        }
        LOG(INFO)<<"cut points val:"<<updater.v_cut[0].cut_points_val;
        LOG(INFO)<<"cut row ptr:"<<updater.v_cut[0].cut_row_ptr;
        //init bin id before the training
        vector<SyncArray<int>> bin_ids(numP);
        for(int i = 0; i < numP; i++){
            updater.init_bin_id_unsort(columns_csc[i], bin_ids[i]);

        }

        LOG(INFO)<<"bin ids 0:"<<bin_ids[0];
        int round = 0;
        //id of company training
        int train_cid = 0;
        bool init_bin = 0;
        for(Tree &tree:trees){
            {
                TIMED_SCOPE(timerObj, "updateGH");
                updateGH_eachtree_naive(round, numP, init_bin, stats, train_cid, n_tree, n_instances);
            }

            LOG(INFO)<<"after update GH";
            if(!init_bin)
            {
                TIMED_SCOPE(timerObj, "init bin id");
                updater.init_bin_id(v_columns[train_cid]);
                init_bin = 1;
                LOG(INFO)<<"updater bin id:"<<(*updater.bin_id[0]);
            }
            {
                TIMED_SCOPE(timerObj, "grow");
                updater.grow(tree, v_columns[train_cid], stats[train_cid]);
            }
            {
                TIMED_SCOPE(timerObj, "prune");
                tree.prune_self(param.gamma);
            }

            LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
            predict_in_training(stats[train_cid], tree);
            for(int i = 0; i < numP; i++) {
                if(i != train_cid)
                    predict_the_company(tree, columns_csc[i], bin_ids[i], param.depth, stats[i], n_instances[i]);
            }
            //next round
            round++;
        }
        DataSet test_dataSet;
        test_dataSet.max_dimension = max_dimension;
        test_dataSet.load_from_file(param.test_path);
        size_t test_instances = test_dataSet.n_instances();
        vector<float_type> y_predict(test_instances, 0);
        predict_test_dataset_rmse_hist(trees, test_dataSet, updater, y_predict);
        int right_num = 0;
        for(int i = 0; i < test_instances; i++){
            if((y_predict[i] > 0.5) == (test_dataSet.y()[i] == 1))
                right_num++;
        }
        float acc = right_num * 1.0 / test_instances;
        LOG(INFO)<<"test error:"<<(1 - acc);

        std::ofstream outfile;
        outfile.open(file_name, std::ios::out | std::ios::app);
        outfile<<"train with "<<numP<<" party:"<<(1-acc)<<std::endl;
        outfile.close();
        return 0;
    }

    void updateGH_eachcompany(const int round, const int numP, bool& init_bin, vector<InsStat>& stats,
            const vector<vector<int>> &similar_id, int& train_cid, const vector<int>& n_tree,
            const vector<int>& n_instances){
        if(round == 0){
            GHPair* stat_gh_host = stats[0].gh_pair.host_data();
            int off = 0;
            for(int cid = 1; cid < numP; cid ++){
                GHPair* stat_gh_host_cid = stats[cid].gh_pair.host_data();
                for(int iid = 0; iid < n_instances[cid]; iid++){
                    stat_gh_host[similar_id[0][off + iid]].g += stat_gh_host_cid[iid].g;
                    stat_gh_host[similar_id[0][off + iid]].h += stat_gh_host_cid[iid].h;
                }
                off += n_instances[cid];
            }
            stats[0].sum_gh.g = 0;
            stats[0].sum_gh.h = 0;
            for(int i = 0; i < n_instances[0]; i++){
                stats[0].sum_gh = stats[0].sum_gh + stat_gh_host[i];
            }
        }
        else{
            stats[train_cid].updateGH();
        }
        if((train_cid != (numP - 1)) && round == n_tree[train_cid+ 1]) {

            GHPair* stat_gh_host = stats[train_cid].gh_pair.host_data();
            int off = 0;
            for(int cid = 0; cid < numP; cid++) {
                //skip the stat just trained
                if (cid == train_cid)
                    continue;
                GHPair *stat_gh_host_cid = stats[cid].gh_pair.host_data();

                for (int iid = 0; iid < n_instances[cid]; iid++) {
                    stat_gh_host_cid[iid].g = stat_gh_host[similar_id[train_cid][off + iid]].g;
                    stat_gh_host_cid[iid].h = stat_gh_host[similar_id[train_cid][off + iid]].h;
                }
                off += n_instances[cid];
                stats[cid].sum_gh.g = 0;
                stats[cid].sum_gh.h = 0;
                for (int i = 0; i < n_instances[cid]; i++) {
                    stats[cid].sum_gh = stats[cid].sum_gh + stat_gh_host_cid[i];
                }
            }

            train_cid++;
            init_bin = 0;
            GHPair* stat_gh_host_new = stats[train_cid].gh_pair.host_data();

            ///update the current stat
            off = 0;
            for(int cid = 0; cid < numP; cid++){
                if(cid == train_cid)
                    continue;
                GHPair* stat_gh_host_cid = stats[cid].gh_pair.host_data();
                for(int iid = 0; iid < n_instances[cid]; iid++){
                    stat_gh_host_new[similar_id[train_cid][off + iid]].g += stat_gh_host_cid[iid].g;
                    stat_gh_host_new[similar_id[train_cid][off + iid]].h += stat_gh_host_cid[iid].h;
                }
                off += n_instances[cid];
            }
            stats[train_cid].sum_gh.g = 0;
            stats[train_cid].sum_gh.h = 0;
            for(int iid = 0; iid < n_instances[train_cid]; iid++){
                stats[train_cid].sum_gh = stats[train_cid].sum_gh + stat_gh_host_new[iid];
            }
        }
        return;
    }


    void updateGH_eachtree(const int round, const int numP, bool& init_bin, vector<InsStat>& stats,
                              const vector<vector<int>> &similar_id, int& train_cid, const vector<int>& n_tree,
                              const vector<int>& n_instances){
        for(int i = 0; i < stats.size(); i++){
            stats[i].updateGH();
        }
        if((train_cid != (numP - 1)) && round == n_tree[train_cid+ 1]){
            train_cid++;
            init_bin = 0;
        }
        int off = 0;
        GHPair* stat_gh_host_new = stats[train_cid].gh_pair.host_data();
        for(int cid = 0; cid < numP; cid++){
            if(cid == train_cid)
                continue;
            GHPair* stat_gh_host_cid = stats[cid].gh_pair.host_data();
            for(int iid = 0; iid < n_instances[cid]; iid++){
                stat_gh_host_new[similar_id[train_cid][off + iid]].g += stat_gh_host_cid[iid].g;
                stat_gh_host_new[similar_id[train_cid][off + iid]].h += stat_gh_host_cid[iid].h;
            }
            off += n_instances[cid];
        }
        stats[train_cid].sum_gh.g = 0;
        stats[train_cid].sum_gh.h = 0;
        for (int i = 0; i < stats[train_cid].gh_pair.size(); i++) {
            stats[train_cid].sum_gh = stats[train_cid].sum_gh + stat_gh_host_new[i];
        }
        return;
    }

    void updateGH_eachtree_exact(const int round, const int numP, vector<InsStat>& stats,
                           const vector<vector<int>> &similar_id, int& train_cid, const vector<int>& n_tree,
                           const vector<int>& n_instances){
        for(int i = 0; i < stats.size(); i++){
            stats[i].updateGH();
        }
        if(train_cid != (numP - 1))
            train_cid++;
        else
            train_cid = 0;
        int off = 0;
        GHPair* stat_gh_host_new = stats[train_cid].gh_pair.host_data();
        for(int cid = 0; cid < numP; cid++){
            if(cid == train_cid)
                continue;
            GHPair* stat_gh_host_cid = stats[cid].gh_pair.host_data();
            for(int iid = 0; iid < n_instances[cid]; iid++){
                stat_gh_host_new[similar_id[train_cid][off + iid]].g += stat_gh_host_cid[iid].g;
                stat_gh_host_new[similar_id[train_cid][off + iid]].h += stat_gh_host_cid[iid].h;
            }
            off += n_instances[cid];
        }
        stats[train_cid].sum_gh.g = 0;
        stats[train_cid].sum_gh.h = 0;
        for (int i = 0; i < stats[train_cid].gh_pair.size(); i++) {
            stats[train_cid].sum_gh = stats[train_cid].sum_gh + stat_gh_host_new[i];
        }
        return;
    }

    void updateGH_eachtree_naive(const int round, const int numP,  vector<InsStat>& stats,
                           int& train_cid, const vector<int>& n_tree,
                           const vector<int>& n_instances){
        for(int i = 0; i < stats.size(); i++){
            stats[i].updateGH();
        }
        if(train_cid != (numP - 1))
            train_cid++;
        else
            train_cid = 0;
        return;
    }

    void updateGH_eachtree_naive(const int round, const int numP, bool &init_bin, vector<InsStat>& stats,
                                 int& train_cid, const vector<int>& n_tree,
                                 const vector<int>& n_instances){
        for(int i = 0; i < stats.size(); i++){
            stats[i].updateGH();
        }
        if((train_cid != (numP - 1)) && round == n_tree[train_cid+ 1]){
            train_cid++;
            init_bin = 0;
        }
        return;
    }


    void predict_the_company(const Tree& tree, SparseColumns& columns,
            SyncArray<int>& bin_id, int tdepth, InsStat &stats, int n_instances){
        const int *iid_data = columns.csc_row_ind.host_data();
        const int *col_ptr_data = columns.csc_col_ptr.host_data();
        const int *bin_id_data = bin_id.host_data();
        int* nid_data = stats.nid.host_data();
        vector<int> nid_data_update(n_instances, 0);
        int n_column = columns.n_column;
        const Tree::TreeNode *nodes_data = tree.nodes.host_data();
        for(int i = 0; i < n_instances; i++)
            nid_data[i] = 0;
        for(int depth = 0; depth < tdepth; depth++){
            int n_max_nodes_in_level = 1 << depth;
            int nid_offset = (1 << depth) - 1;
            for(int iid = 0; iid < n_instances; iid++)
                nid_data_update[iid] = nid_data[iid];
            for(int col_id = 0; col_id < n_column; col_id++){
                for(int fid = col_ptr_data[col_id]; fid < col_ptr_data[col_id + 1]; fid++){
                    int iid = iid_data[fid];
                    int nid = nid_data[iid];
                    const Tree::TreeNode &node = nodes_data[nid];
                    if((node.splittable()) && (node.split_feature_id == col_id)){
                        if((float_type) bin_id_data[fid] > node.split_value)
                            nid_data_update[iid] = node.lch_index;
                        else
                            nid_data_update[iid] = node.rch_index;
                    }
                }
            }
            for(int iid = 0; iid < n_instances; iid++)
                nid_data[iid] = nid_data_update[iid];
            for(int iid = 0; iid < n_instances; iid++){
                int nid = nid_data[iid];
                if(nodes_data[nid].splittable() && (nid < nid_offset + n_max_nodes_in_level)){
                    const Tree::TreeNode &node = nodes_data[nid];
                    if(node.default_right)
                        nid_data[iid] = node.rch_index;
                    else
                        nid_data[iid] = node.lch_index;
                }
            }
        }
        float_type* y_predict_data = stats.y_predict.host_data();
        for(int iid = 0; iid < n_instances; iid++){
            int nid = nid_data[iid];
            while(nid != -1 && (nodes_data[nid].is_pruned))
                nid = nodes_data[nid].parent_index;
            y_predict_data[iid] += nodes_data[nid].base_weight;
        }
    }

    void predict_the_company_exact(const Tree& tree, SparseColumns& columns,
                             int tdepth, InsStat &stats, int n_instances){
        const int *iid_data = columns.csc_row_ind.host_data();
        const int *col_ptr_data = columns.csc_col_ptr.host_data();
        const float_type *csc_val_data = columns.csc_val.host_data();
        int* nid_data = stats.nid.host_data();
        vector<int> nid_data_update(n_instances, 0);
        int n_column = columns.n_column;
        const Tree::TreeNode *nodes_data = tree.nodes.host_data();
        for(int i = 0; i < n_instances; i++)
            nid_data[i] = 0;
        for(int depth = 0; depth < tdepth; depth++){
            int n_max_nodes_in_level = 1 << depth;
            int nid_offset = (1 << depth) - 1;
            for(int iid = 0; iid < n_instances; iid++)
                nid_data_update[iid] = nid_data[iid];
            for(int col_id = 0; col_id < n_column; col_id++){
                for(int fid = col_ptr_data[col_id]; fid < col_ptr_data[col_id + 1]; fid++){
                    int iid = iid_data[fid];
                    int nid = nid_data[iid];
                    const Tree::TreeNode &node = nodes_data[nid];
                    if((node.splittable()) && (node.split_feature_id == col_id)){
                        if(csc_val_data[fid] < node.split_value)
                            nid_data_update[iid] = node.lch_index;
                        else
                            nid_data_update[iid] = node.rch_index;
                    }
                }
            }
            for(int iid = 0; iid < n_instances; iid++)
                nid_data[iid] = nid_data_update[iid];
            for(int iid = 0; iid < n_instances; iid++){
                int nid = nid_data[iid];
                if(nodes_data[nid].splittable() && (nid < nid_offset + n_max_nodes_in_level)){
                    const Tree::TreeNode &node = nodes_data[nid];
                    if(node.default_right)
                        nid_data[iid] = node.rch_index;
                    else
                        nid_data[iid] = node.lch_index;
                }
            }
        }
        float_type* y_predict_data = stats.y_predict.host_data();
        for(int iid = 0; iid < n_instances; iid++){
            int nid = nid_data[iid];
            while(nid != -1 && (nodes_data[nid].is_pruned))
                nid = nodes_data[nid].parent_index;
            y_predict_data[iid] += nodes_data[nid].base_weight;
        }
    }
};

class PerformanceTest : public UpdaterTest {
};

void parse_command_line(){
    for (int i = 1; i < iargc; i++){
        if (iargv[i][0] != '-')
            break;
        if (++i >= iargc)
            printf("wrong input format");
        switch (iargv[i-1][1]){
            case 'p':
                numP = atoi(iargv[i]);
                break;
            case 't':
                n_table = atoi(iargv[i]);
                break;
            case 'b':
                n_bucket = atoi(iargv[i]);
                break;
            case 'r':
                r = atof(iargv[i]);
                break;
            case 'f':
                dataset_path = iargv[i];
                break;
            case 's':
                lsh_seed = atoi(iargv[i]);
                break;
            case 'd':
                max_dim = atoi(iargv[i]);
                break;
            case 'n':
                n_trees = atoi(iargv[i]);
                break;
            case 'e':
                n_depth = atoi(iargv[i]);
                break;
        }
    }
}



TEST_F(PerformanceTest, SimFL) {
    parse_command_line();
    param.n_trees = n_trees;
    param.depth = n_depth;
    std::ofstream outfile;
    outfile.open(file_name, std::ios::out | std::ios::app);
    outfile<<std::endl;
    outfile<<dataset_name<<" "<<"numP:"<<numP<<" "<<"depth"<<param.depth<<" tree"<<param.n_trees<<std::endl;
    outfile<<"r:"<<r<<" n_table:"<<n_table<<" n_bucket:"<<n_bucket<<std::endl;
    outfile.close();
    param.path = dataset_path+"_train";
    param.test_path = dataset_path+"_test";
    train_hist_multi_party_new(param, numP, r, n_table, n_bucket, max_dim);
}
