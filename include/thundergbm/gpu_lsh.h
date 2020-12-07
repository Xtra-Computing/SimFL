#ifndef THUNDERGBM_GPU_LSH_H
#define THUNDERGBM_GPU_LSH_H


//////////////////////////////////////////////////////////////////////////////
/// Copyright (C) 2014 Gefu Tang <tanggefu@gmail.com>. All Rights Reserved.
///
/// This file is part of LSHBOX.
///
/// LSHBOX is free software: you can redistribute it and/or modify it under
/// the terms of the GNU General Public License as published by the Free
/// Software Foundation, either version 3 of the License, or(at your option)
/// any later version.
///
/// LSHBOX is distributed in the hope that it will be useful, but WITHOUT
/// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
/// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
/// more details.
///
/// You should have received a copy of the GNU General Public License along
/// with LSHBOX. If not, see <http://www.gnu.org/licenses/>.
///
/// @version 0.1
/// @author Gefu Tang & Zhifeng Xiao
/// @date 2014.6.30
//////////////////////////////////////////////////////////////////////////////

/**
 * @file psdlsh.h
 *
 * @brief Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.
 */
//#pragma once

#include "thundergbm.h"
#include "cusparse.h"
#include <map>
#include <random>
#include <iostream>
#include <functional>
#include "syncarray.h"
/**
 * Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.
 *
 *
 * For more information on p-stable distribution based LSH, see the following reference.
 *
 *     Mayur Datar , Nicole Immorlica , Piotr Indyk , Vahab S. Mirrokni,
 *     Locality-sensitive hashing scheme based on p-stable distributions,
 *     Proceedings of the twentieth annual symposium on Computational geometry, June
 *     08-11, 2004, Brooklyn, New York, USA.
 */
//template<typename DATATYPE = float>
class psdLsh
{
public:
    struct Parameter
    {
        //number of buckets in a hash table
        unsigned n_bucket;
        //number of hash tables
        unsigned n_table;
        //number of dimensions
        unsigned n_dimension;
        //p_norm = 1, cauchy distribution, = 2, gaussian distribution
        unsigned p_norm;
        //the range r
        float r;
        //the number of companies
        unsigned n_comp;
        int seed;
    };
    psdLsh() {}
    psdLsh(const Parameter &param_)
    {
        reset(param_);
    }
    ~psdLsh() {}


    void init();
    /**
     * Reset the parameter setting
     *
     * @param param_ A instance of psdLsh<DATATYPE>::Parametor, which contains
     * the necessary parameters
     */
    void reset(const Parameter &param_);
    /**
     * Hash the dataset.
     *
     * @param data A instance of Matrix<DATATYPE>, it is the search dataset.
     */
    void hash(int n_instances, int n_features, int nnz, int key_offset,
            SyncArray<float_type> &csr_val, SyncArray<int> &csr_row_ptr, SyncArray<int> &csr_col_ind,
            SyncArray<int> &hash_values, int cid);

//    void query(int n_instances, int n_features, int nnz,
//              SyncArray<float_type> &csr_val, SyncArray<int> &csr_row_ptr, SyncArray<int> &csr_col_ind,
//              vector<vector<int>> &buckets);

    /**
     * Insert a vector to the index.
     *
     * @param key   The sequence number of vector
     * @param domin The pointer to the vector
     */
//        void insert(unsigned key, const DATATYPE *domin);
    /**
     * Query the approximate nearest neighborholds.
     *
     * @param domin   The pointer to the vector
     * @param scanner Top-K scanner, use for scan the approximate nearest neighborholds
     */
//        template<typename SCANNER>
//        void query(const DATATYPE *domin, SCANNER &scanner);
    /**
     * get the hash value of a vector.
     *
     * @param k     The idx of the table
     * @param domin The pointer to the vector
     * @return      The hash value
     */
//        unsigned getHashVal(unsigned k, const DATATYPE *domin);
    /**
     * Load the index from binary file.
     *
     * @param file The path of binary file.
     */
//        void load(const std::string &file);
    /**
     * Save the index as binary file.
     *
     * @param file The path of binary file.
     */
//        void save(const std::string &file);
//private:
    Parameter param;
    SyncArray<float> b;
    SyncArray<float> a;

    vector<float> v_a;
    vector<float> v_b;
    //random number vector before distribution transform
    vector<float> random_vector;


//        std::vector<float> v_a;
//        std::vector<float> rndBs;
//        std::vector<std::vector<float> > stableArray;
    std::vector<std::map<unsigned, std::vector<std::vector<unsigned>> > > tables;
//    std::vector<std::map<int, std::vector<std::vector<unsigned>> > > tables;
};

// ------------------------- implementation -------------------------

//template<typename DATATYPE>
//void lshbox::psdLsh<DATATYPE>::hash(Matrix<DATATYPE> &data)
//{
//    for (unsigned i = 0; i != data.getSize(); ++i)
//    {
//        insert(i, data[i]);
//        ++pd;
//    }
//}
//template<typename DATATYPE>
//void lshbox::psdLsh<DATATYPE>::insert(unsigned key, const DATATYPE *domin)
//{
//    for(unsigned k = 0; k < param.n_table; k++){
//        unsigned  hashVal = getHashVal(k, )
//    }
////    for (unsigned k = 0; k != param.L; ++k)
////    {
////        unsigned hashVal = getHashVal(k, domin);
////        tables[k][hashVal].push_back(key);
////    }
//}
//template<typename DATATYPE>
//template<typename SCANNER>
//void lshbox::psdLsh<DATATYPE>::query(const DATATYPE *domin, SCANNER &scanner)
//{
//    scanner.reset(domin);
//    for (unsigned k = 0; k != param.L; ++k)
//    {
//        unsigned hashVal = getHashVal(k, domin);
//        if (tables[k].find(hashVal) != tables[k].end())
//        {
//            for (std::vector<unsigned>::iterator iter = tables[k][hashVal].begin(); iter != tables[k][hashVal].end(); ++iter)
//            {
//                scanner(*iter);
//            }
//        }
//    }
//    scanner.topk().genTopk();
//}
//template<typename DATATYPE>
//unsigned lshbox::psdLsh<DATATYPE>::getHashVal(unsigned k, const DATATYPE *domin)
//{
//    float sum(0);
//    for (unsigned i = 0; i != param.D; ++i)
//    {
//        sum += domin[i] * stableArray[k][i];
//    }
//    unsigned hashVal = unsigned(std::floor((sum + rndBs[k]) / param.W)) % param.M;
//    return hashVal;
//}
//template<typename DATATYPE>
//void lshbox::psdLsh<DATATYPE>::load(const std::string &file)
//{
//    std::ifstream in(file, std::ios::binary);
//    in.read((char *)&param.M, sizeof(unsigned));
//    in.read((char *)&param.L, sizeof(unsigned));
//    in.read((char *)&param.D, sizeof(unsigned));
//    in.read((char *)&param.W, sizeof(float));
//    tables.resize(param.L);
//    stableArray.resize(param.L);
//    rndBs.resize(param.L);
//    in.read((char *)&rndBs[0], sizeof(float) * param.L);
//    for (unsigned i = 0; i != param.L; ++i)
//    {
//        stableArray[i].resize(param.D);
//        in.read((char *)&stableArray[i][0], sizeof(float) * param.D);
//        unsigned count;
//        in.read((char *)&count, sizeof(unsigned));
//        for (unsigned j = 0; j != count; ++j)
//        {
//            unsigned target;
//            in.read((char *)&target, sizeof(unsigned));
//            unsigned length;
//            in.read((char *)&length, sizeof(unsigned));
//            tables[i][target].resize(length);
//            in.read((char *) & (tables[i][target][0]), sizeof(unsigned) * length);
//        }
//    }
//    in.close();
//}
//template<typename DATATYPE>
//void lshbox::psdLsh<DATATYPE>::save(const std::string &file)
//{
//    std::ofstream out(file, std::ios::binary);
//    out.write((char *)&param.M, sizeof(unsigned));
//    out.write((char *)&param.L, sizeof(unsigned));
//    out.write((char *)&param.D, sizeof(unsigned));
//    out.write((char *)&param.W, sizeof(float));
//    out.write((char *)&rndBs[0], sizeof(float) * param.L);
//    for (int i = 0; i != param.L; ++i)
//    {
//        out.write((char *)&stableArray[i][0], sizeof(float) * param.D);
//        unsigned count = unsigned(tables[i].size());
//        out.write((char *)&count, sizeof(unsigned));
//        for (std::map<unsigned, std::vector<unsigned> >::iterator iter = tables[i].begin(); iter != tables[i].end(); ++iter)
//        {
//            unsigned target = iter->first;
//            out.write((char *)&target, sizeof(unsigned));
//            unsigned length = unsigned(iter->second.size());
//            out.write((char *)&length, sizeof(unsigned));
//            out.write((char *) & ((iter->second)[0]), sizeof(unsigned) * length);
//        }
//    }
//    out.close();
//}


#endif //THUNDERGBM_GPU_LSH_H
