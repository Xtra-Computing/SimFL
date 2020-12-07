//
// Created by jiashuai on 18-1-17.
//

#ifndef THUNDERGBM_DATASET_H
#define THUNDERGBM_DATASET_H

#include "thundergbm.h"
#include "syncarray.h"

class DataSet {
public:
    ///one feature value and corresponding index
    struct node {
        node(int index, float_type value) : index(index), value(value) {}

        int index;
        float_type value;
    };

    ///two-dimension node vector
    typedef vector<vector<DataSet::node>> node2d;

    ///load dataset from file
    void load_from_file(string file_name);

    void load_from_file_csr(string file_name);

    void load_from_file_two_dimension(string file_name);

    const node2d &instances() const;

    size_t n_features() const;

    size_t n_instances() const;

    const vector<float_type> &y() const;
	vector<vector<float_type>> features;
    vector<vector<int>> line_num;

    //to do
    vector<float> min_fea;
    vector<float> max_fea;

    //suppose know the max dimension of all datasets
    int max_dimension = 123;

    void compression();
private:
    ///labels of instances
    vector<float_type> y_;
    node2d instances_;
    size_t n_features_;
};


#endif //THUNDERGBM_DATASET_H
