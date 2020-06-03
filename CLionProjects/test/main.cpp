/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using namespace tensorflow;
using namespace tensorflow::ops;
#include "LR_model.h"

int main(int argc, char* argv[]) {
    srand(0);
    LR_model tes;
    /*string folder_name;
    vector<pair<string,float>> v_folder_label;
    int batch_size;
    vector<Tensor> image_batches;
    vector<Tensor> label_batches;
    tes.ReadBatches(folder_name,v_folder_label,batch_size,image_batches,label_batches);*/

    Status s;
    s=tes.CreateGraphForData();
    TF_CHECK_OK(s);
    string file_name = "/home/hqf/CLionProjects/train.txt";
    vector<Tensor> image_batches, label_batches;
    tes.ReadBatches(file_name,20,image_batches,label_batches);

    s=tes.CreateGraphForLR(100,1,0.01);
    TF_CHECK_OK(s);
    s = tes.TrainLR(image_batches,label_batches,10000);
    TF_CHECK_OK(s);
    //tes.TestDataReader();
    //tes.CreateGraphForLR();
    //tes.TestLRGraph();
    return 0;
}
