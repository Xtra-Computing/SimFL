//
// Created by jiashuai on 17-9-15.
//
#include "thundergbm/thundergbm.h"
#include "gtest/gtest.h"

//#include <thundergbm/tree.h>
//#include <thundergbm/dataset.h>
//#include <thundergbm/updater/exact_updater.h>
//#include <thundergbm/updater/hist_updater.h>


//float_type compute_rmse(const InsStat &stats) {
//    float_type sum_error = 0;
//    const float_type *y_data = stats.y.host_data();
//    const float_type *y_predict_data = stats.y_predict.host_data();
//    for (int i = 0; i < stats.n_instances; ++i) {
//        float_type e = y_predict_data[i] - y_data[i];
//        sum_error += e * e;
//    }
//    float_type rmse = sqrt(sum_error / stats.n_instances);
//    return rmse;
//}



int iargc; // Making arg and arv global to access within TESTs
char** iargv;
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    iargc = argc;
    iargv = argv;
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
    el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
    return RUN_ALL_TESTS();

//    GBMParam param;
//    bool verbose = false;
//    param.depth = 6;
//    param.n_trees = 40;
//    param.min_child_weight = 1;
//    param.lambda = 1;
//    param.gamma = 1;
//    param.rt_eps = 1e-6;
//    param.do_exact = true;
//    param.n_device = 1;
//    if (!verbose) {
//        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
//        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
//    }
//    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "true");
//    DataSet dataSet;
//    dataSet.load_from_file(param.path);
//    int n_instances = dataSet.n_instances();
//    InsStat stats;
//    vector<Tree> trees;
//    SparseColumns columns;
//    columns.from_dataset(dataSet);
//    trees.resize(param.n_trees);
//    stats.resize(n_instances);
//    stats.y.copy_from(dataSet.y().data(), n_instances);
//
//    int n_devices = 1;
//    vector<std::shared_ptr<SparseColumns>> v_columns;
//    v_columns.resize(n_devices);
//    for (int i = 0; i < n_devices; i++)
//        v_columns[i].reset(new SparseColumns());
//    columns.to_multi_devices(v_columns);
//    HistUpdater updater(param);
//    int round = 0;
//    float_type rmse = 0;
//    {
//        bool init_bin = 0;
//        for (Tree &tree:trees) {
//            stats.updateGH();
//            //updater.insBundle(v_columns, stats);
//            TIMED_SCOPE(timerObj, "construct tree");
//            if(!init_bin) {
//                updater.use_similar_bundle = 0;
//                {
//                    TIMED_SCOPE(timerObj, "init cut");
//                    updater.init_cut(v_columns, stats, n_instances, columns);
//                }
//                if(updater.use_similar_bundle)
//                {
//                    TIMED_SCOPE(timerObj, "similar ins bundle");
//                    updater.similar_ins_bundle(v_columns, stats, n_instances, dataSet, columns);
//                }
//                init_bin = 1;
//            }
//
//
//            {
//                TIMED_SCOPE(timerObj, "grow");
//                updater.grow(tree, v_columns, stats);
//            }
//            {
//                TIMED_SCOPE(timerObj, "prune");
//                tree.prune_self(param.gamma);
//            }
//
//            LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
//            predict_in_training(stats, tree);
//            //next round
//            round++;
//
//        }
//    }
//    rmse = compute_rmse(stats);
//    LOG(INFO) << "rmse = " << rmse;
//    return 1;
}
