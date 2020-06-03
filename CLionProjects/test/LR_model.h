//
// Created by root on 2020/5/25.
//

#ifndef TEST_LR_MODEL_H
#define TEST_LR_MODEL_H

#endif //TEST_LR_MODEL_H
#include <iostream>
#include <map>
#include <fstream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
class LR_model{
private:
    Output weight;
    Output b;
    Output assign_w;
    Output assign_b;
    Output apply_w;
    Output apply_b;
    Scope d_root;//graph for loading data into tensors
    Scope t_root;//training and validating the LR
    unique_ptr<ClientSession> t_session;
    Output file_name_var;
    Output file_reader;
    Output file_tensor;
    Output D_file_tensor;
    Output file_num_tensor;
    vector<string> all_file_number;
    Output input_batch_var;
    string input_name = "input";
    Output input_labels_var;
    Output drop_rate_var;
    string drop_rate_name = "drop_rate";
    Output skip_drop_var;
    string skip_drop_name = "skip_drop";
    Output out_classification;
    Output dim;
    string out_name = "output_classes";
    Output logits;
    string logits_name = "logits";
    //Network maps
    map<string,Output> m_vars;
    //Loss variables;
    vector<Output> v_weights_biases;
    vector<Operation> v_out_grads;
    Output out_loss_var;
    InputList MakeTransforms(int batch_size,Input a0,Input a1,Input a2,Input b0,Input b1,Input b2);

public:
    LR_model() : d_root(Scope::NewRootScope()), t_root(Scope::NewRootScope()) {}
    Status CreateGraphForData();
    Status TestDataReader();
    Status ReadFileTensors(string& folder_name, vector<pair<string, float>> v_folder_label, vector<pair<Tensor, float>>& file_tensors,int batch_size);
    Status ReadBatches(string& folder_name, int batch_size, vector<Tensor>& image_batches, vector<Tensor>& label_batches);
    Status CreateGraphForLR(int in_channels,int out_channels,float learn_rate);
    Status TestLRGraph();
    Status Initialize();
    Status TrainLR(vector<Tensor>& image_batchs, vector<Tensor>& label_batchs,int times);
    Status ValidateLR(Tensor& image_batch, Tensor& label_batch, vector<float>& results);
    Status Predict(Tensor& image, int& result);
    Status FreezeSave(string& file_name);
    Status LoadSavedModel(string& file_name);
    Status PredictFromFrozen(Tensor& image, int& result);
    Status CreateAugmentGraph(int batch_size, int image_side, float flip_chances, float max_angles, float sscale_shift_factor);
    Status RandomAugmentBatch(Tensor& image_batch, Tensor& augmented_batch);
    Status WriteBatchToImageFiles(Tensor& image_batch, string folder_name, string image_name);

};