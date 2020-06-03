//
// Created by root on 2020/5/25.
//

#include "LR_model.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
Status LR_model::CreateGraphForData(){
    file_name_var = Placeholder(d_root.WithOpName("input"),DT_STRING);
    file_reader = ReadFile(d_root.WithOpName("file_reader"),file_name_var);
    auto reshape_file_reader = Reshape(d_root,file_reader,{1});
    //file_reader = DecodeCompressed(d_root.WithOpName("de_comp"),file_reader);
    //file_reader = TextLineReader(d_root);
    string delim="\n";
    //string dou=",";
    string mao = ":";
    string empty = " ";
    auto ori_file_tensor = StringSplit(d_root.WithOpName("all_number"),reshape_file_reader,delim).values;
    //file_tensor = Reshape(d_root,ori_file_tensor,{-1,1});
    auto DD_file_tensor = StringSplit(d_root.WithOpName("2D_all_number"),ori_file_tensor,mao).values;
    D_file_tensor = StringSplit(d_root,DD_file_tensor,empty).values;
    file_num_tensor = StringToNumber(d_root,D_file_tensor);
    //float_caster = Cast(d_root.WithOpName("float_caster"), file_reader, DT_FLOAT);
    //SpaceToBatch(d_root.WithOpName("StoN"),1,2,file_reader,{1,-1},2);
    //file_number = DecodeCSV(d_root.WithOpName("csv_reader"), file_reader, {{delim},{dou}},DecodeCSV::UseQuoteDelim(false)).output;
    return d_root.status();
}
Status LR_model::ReadBatches(string &file_name, int batch_size, vector<Tensor> &image_batches, vector<Tensor> &label_batches) {
    vector<pair<Tensor,float>> all_file_tensors;
    vector<Tensor> out_tensors;
    ClientSession session(d_root);
    TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {file_num_tensor}, &out_tensors));
    Tensor outTensor = out_tensors[0];
    auto a = outTensor.tensor<float,1>();
    TensorShape shape = outTensor.shape();
    int output_dim = shape.dim_size(0);
    vector<vector<float>> x_;
    vector<float> tem;
    vector<float> label;
    int num=0;
    for(int i=0;i<output_dim;i++){
        label.push_back(a(i));
        num++;
        int now=0;
        i++;
        int x_now=1;
        while(now<a(i)){
            now=a(i);
            while(x_now<now){
                tem.push_back(0);
                x_now++;
            }
            tem.push_back(1);
            x_now++;
            i+=2;
        }
        while(x_now<121){
            tem.push_back(0);
            x_now++;
        }
        i--;
        if(label.size()==batch_size) {
            Tensor X_(DataTypeToEnum<float>::v(),TensorShape{batch_size,120});
            copy_n(tem.begin(),tem.size(),X_.flat<float>().data());
            image_batches.push_back(X_);
            Tensor y_(DataTypeToEnum<float>::v(),TensorShape{1,batch_size});
            copy_n(label.begin(),label.size(),y_.flat<float>().data());
            label_batches.push_back(y_);
            tem.clear();label.clear();
        }
    }
    return d_root.status();
}
Status LR_model::CreateGraphForLR(int in_channels,int out_channels,float learn_rate)
{
    Placeholder::Attrs input_batch_var_shape = Placeholder::Attrs().Shape(TensorShape({20,120}));
    input_batch_var = Placeholder(t_root.WithOpName(input_name), DT_FLOAT, input_batch_var_shape);
    input_labels_var = Placeholder(t_root,DT_FLOAT);
    weight = Variable(t_root,{1,120},DT_FLOAT);
    assign_w = Assign(t_root,weight,RandomNormal(t_root,{1,120},DT_FLOAT));
    b = Variable(t_root,{1,20},DT_FLOAT);
    assign_b = Assign(t_root,b,RandomNormal(t_root,{1,20},DT_FLOAT));
    //forward
    auto one = OnesLike(t_root,input_labels_var);
    logits = Add(t_root,MatMul(t_root.WithOpName(logits_name),weight,Transpose(t_root,input_batch_var,{1,0})),b);
    out_classification = Sigmoid(t_root.WithOpName(out_name), logits);
    //out_loss_var = Negate(t_root,ReduceMean(t_root,Transpose(t_root,Log(t_root,Sub(t_root,input_labels_var,Sub(t_root,one,out_classification))),{1,0}),{0,1}));
    out_loss_var =ReduceMean(t_root,Negate(t_root,Add(t_root,Multiply(t_root,input_labels_var,Log(t_root,out_classification)),
                       Multiply(t_root,Sub(t_root,one,input_labels_var),Log(t_root,Sub(t_root,one,out_classification))))),{0,1});
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(t_root, {out_loss_var}, {weight, b}, &grad_outputs));
    apply_w = ApplyGradientDescent(t_root,weight,Cast(t_root,learn_rate,DT_FLOAT),{grad_outputs[0]});
    apply_b = ApplyGradientDescent(t_root,b,Cast(t_root,learn_rate,DT_FLOAT),{grad_outputs[1]});
    return t_root.status();
}
Status LR_model::TrainLR(vector<Tensor>& image_batchs, vector<Tensor>& label_batchs,int ti)
{
    if(!t_root.ok())
        return t_root.status();

    vector<Tensor> out_tensors;
    //Inputs: batch of images, labels, drop rate and do not skip drop.
    //Extract: Loss and result. Run also: Apply Adam commands
    Tensor image_batch(DT_FLOAT, TensorShape({20, 120}));
    Tensor label_batch(DT_FLOAT, TensorShape({1, 20}));
    t_session = unique_ptr<ClientSession>(new ClientSession(t_root));
    TF_CHECK_OK(t_session->Run({assign_w,assign_b},nullptr));
    for(int i=1;i<=ti;i++){
        for(int j=0;j<1;j++) {
            image_batch = image_batchs[j];
            label_batch = label_batchs[j];
            TF_CHECK_OK(t_session->Run({{input_batch_var,  image_batch},
                                             {input_labels_var, label_batch}}, {out_loss_var,out_classification},
                                        &out_tensors));
            if(j==0&&i%100==0){
                cout<<out_tensors[0].flat<float>()<<"\n";
            }
            TF_CHECK_OK(t_session->Run({{input_batch_var,  image_batch},
                                        {input_labels_var, label_batch}},{apply_w,apply_b}, nullptr));
        }
    }
    return Status::OK();
}
Status LR_model::TestDataReader(){ //test ok!
    string file_name = "/home/hqf/CLionProjects/train.txt";
    vector<Tensor> out_tensors;
    ClientSession session(d_root);
    cout << " get into function \n";
    TF_CHECK_OK(session.Run({{file_name_var, file_name}},{file_tensor}, &out_tensors));
    cout<< out_tensors[0].DebugString()<<'\n';
    return Status::OK();
}

Status LR_model::TestLRGraph(){ //
    vector<Tensor> out_tensors;
    ClientSession session(t_root);
    cout << " get into function \n";
    Tensor batch(DT_FLOAT,TensorShape({2, 1}));
    TF_CHECK_OK(session.Run({{input_batch_var,batch}},{dim,out_classification},&out_tensors));
    cout<< out_tensors[0].DebugString()<<'\n';
    return Status::OK();
}
