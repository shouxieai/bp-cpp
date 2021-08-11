
#include "matrix.hpp"
#include <algorithm>
#include <fstream>

using namespace std;

// 定义全局的随机数引擎
static default_random_engine global_random_engine;


/**
 * @brief 创建一个正太分布的随机数矩阵
 * 
 * @param rows    矩阵的行数
 * @param cols    矩阵的列数
 * @param mean    正太分布随机数的均值
 * @param stddev  正太分布随机数的标准差
 * @return Matrixf 
 */
Matrixf create_normal_distribution_matrix(int rows, int cols, float mean=0.0f, float stddev=1.0f){

    normal_distribution<float> norm(mean, stddev);
    Matrixf out(rows, cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j)
            out.ptr(i)[j] = norm(global_random_engine);
    }
    return out;
}

/**
 * @brief 填充一个伯努利分布的随机数矩阵
 * 
 * @param prob  指定为1的概率
 */
void fill_bernoulli_distribution_matrix(Matrixu& matrix, float prob = 0.5){

    bernoulli_distribution bernoulli(prob);
    auto ptr = matrix.ptr();
    for(int i = 0; i < matrix.numel(); ++i, ++ptr){
        *ptr = bernoulli(global_random_engine);
    }
}

/**
 * @brief 加载mnist数据集的label文件
 *  如果文件不符合返回空矩阵
 * 
 * @param file   mnist的label文件
 * @return Matrixu 
 */
Matrixu load_mnist_labels(const string& file){

    Matrixu out;
    ifstream in(file, ios::binary | ios::in);

    if(!in.is_open()){
        INFO("open %s failed", file.c_str());
        return out;
    }

    int header[2];
    in.read((char*)header, sizeof(header));

    if(InvertBit(header[0]) != 0x00000801){
        INFO("%s is not nmist labels file", file.c_str());
        return out;
    }

    int num_labels = InvertBit(header[1]);
    out.resize(num_labels, 1);
    in.read((char*)out.ptr(), num_labels);
    return out;
}

/**
 * @brief 加载mnist数据集的图像文件
 * 
 * @param file   mnist的图像文件
 * @return Matrixu 
 */
Matrixu load_mnist_images(const string& file){

    Matrixu out;
    ifstream in(file, ios::binary | ios::in);
    
    if(!in.is_open()){
        INFO("open %s failed", file.c_str());
        return out;
    }

    int header[4];
    in.read((char*)header, sizeof(header));

    if(InvertBit(header[0]) != 0x00000803){
        INFO("%s is not nmist images file", file.c_str());
        return out;
    }

    int num_images  = InvertBit(header[1]);
    int rows        = InvertBit(header[2]);
    int cols        = InvertBit(header[3]);
    out.resize(num_images, rows * cols);
    in.read((char*)out.ptr(), out.numel());
    return out;
}

/**
 * @brief 加载一个bmp的图像文件为矩阵
 * 请提供28x28的bmp图像文件。这里只加载28x28的图抱歉哈
 * 
 * @param file bmp图像文件，必须是28x28的24bit文件
 * @return Matrixu 
 */
Matrixu load_bmp_matrix(const string& file){

    ifstream in(file, ios::binary|ios::in);
    if(!in.is_open()){
        INFO("Open %s failed", file.c_str());
        return Matrixu();
    }

    Matrixu gray(28, 28);
    Matrixu rgb(28, 28 * 3);

    // BMP的48字节是头
    in.seekg(48, ios::cur);

    // 之后是RGB的图像数据
    in.read((char*)rgb.ptr(), rgb.numel());

    // BMP的数据起点是左下角，因此需要颠倒
    for(int i = 0; i < gray.rows(); ++i){
        auto pgray = gray.ptr(i);
        auto prgb = rgb.ptr(rgb.rows() - 1 - i);
        for(int j = 0; j < gray.cols(); ++j)
            pgray[j] = prgb[j * 3];
    }

    if(!in.good())
        return Matrixu();

    // 把加载的数据弄成1行
    gray.resize(1, 784);
    return gray;
}

/**
 * @brief 归一化图像到数据，把uint8格式转换为float
 * 并且进行归一化，即：dst = src / 255.0f - 0.5f;
 * 
 * @param images 输入的图像数据，是uint8的矩阵
 * @return Matrixf 
 */
Matrixf normalize_image_to_matrix(const Matrixu& images){

    Matrixf out(images.rows(), images.cols());
    auto out_ptr = out.ptr();
    auto image_ptr = images.ptr();
    for(int i = 0; i < images.numel(); ++i, ++out_ptr, ++image_ptr)
        //*out_ptr = *image_ptr / 255.0f - 0.5f;
        *out_ptr = (*image_ptr / 255.0f - 0.1307f) / 0.3081f;
    return out;
}

/**
 * @brief 选择矩阵的某些行
 * 返回的新矩阵，是根据indexs为索引数组，以begin为起点，size为次数，获取特定行
 * dst = m[indexs[begin:begin + size]]
 * 
 * @param m       取值的矩阵m
 * @param indexs  索引值数组
 * @param begin   起点，即row = indexs[begin]            为第一个获取的行
 * @param size    大小，即row = indexs[begin + size - 1] 为最后一个获取的行
 * @return Matrix<_DataType> 
 */
template<typename _DataType>
Matrix<_DataType> choice_rows(const Matrix<_DataType>& m, const vector<int>& indexs, int begin=0, int size=-1){

    if(size == -1) size = indexs.size();
    Matrix<_DataType> out(size, m.cols());
    for(int i = 0; i < size; ++i){
        int mrow = indexs[i + begin];
        int orow = i;
        memcpy(out.ptr(orow), m.ptr(mrow), sizeof(_DataType) * m.cols());
    }
    return out;
}

/**
 * @brief 将label转换到onehot，即，对于10个类别，label = 2时
 * 返回值为：0, 0, 1, 0, 0, 0, 0, 0, 0, 0
 * 
 * @param labels       标签矩阵，num x 1的矩阵
 * @param num_classes  类别数
 * @return Matrixf 
 */
Matrixf label_to_onehot(const Matrixu& labels, int num_classes=10){

    Matrixf out(labels.rows(), num_classes);
    for(int i = 0; i < out.rows(); ++i)
        out.ptr(i)[labels.ptr(i)[0]] = 1;
    return out;
}

/**
 * @brief 对y = sigmoid(x)中，对x求导
 * 输入的value = sigmoid(x)
 * sigmoid求导结果是 = sigmoid(x) * (1 - sigmoid(x))
 * 因此这里计算的是  = value * (1 - value)
 * 
 * @param sigmoid_value 输入的是sigmoid(x)
 * @return Matrixf 
 */
Matrixf delta_sigmoid(const Matrixf& sigmoid_value){
    auto out = sigmoid_value.copy();
    auto ptr = out.ptr();
    for(int i = 0; i < out.numel(); ++i, ++ptr)
        *ptr = *ptr * (1 - *ptr);
    return out;
}

/**
 * @brief 对y = relu(x)中，对x求导
 * f' = x <= 0 ? 0 : 1
 * 
 * @param grad       loss对y的导数
 * @param x          x的值
 * @return Matrixf 
 */
Matrixf delta_relu(const Matrixf& grad, const Matrixf& x){
    auto out = grad.copy();
    auto optr = out.ptr();
    auto xptr = x.ptr();
    for(int i = 0; i < out.numel(); ++i, ++optr, ++xptr){
        if(*xptr <= 0)
            *optr = 0;
    }
    return out;
}

/**
 * @brief 行方向求和
 * 对矩阵的所有行求和，即dst[1xn] = value[mxn]
 * 所有行求和变为1行
 * 
 * @param value 给定的求和矩阵
 * @return Matrixf 
 */
Matrixf row_sum(const Matrixf& value){
    Matrixf out(1, value.cols());
    auto optr = out.ptr();
    auto vptr = value.ptr();
    for(int i = 0; i < value.numel(); ++i, ++vptr)
        optr[i % value.cols()] += *vptr;
    return out;
}

/**
 * @brief 计算Sigmoid交叉熵loss
 * loss = sum(-(y * log(p) + (1 - y) * log(1 - p))) / N
 * 
 * @param probability   预测的概率值 
 * @param onehot_labels 真值onehot格式
 * @return float 
 */
float compute_loss(const Matrixf& probability, const Matrixf& onehot_labels){

    float eps = 1e-5;
    float sum_loss  = 0;
    auto pred_ptr   = probability.ptr();
    auto onehot_ptr = onehot_labels.ptr();
    int numel       = probability.numel();
    for(int i = 0; i < numel; ++i, ++pred_ptr, ++onehot_ptr){
        auto y = *onehot_ptr;
        auto p = *pred_ptr;
        p = max(min(p, 1 - eps), eps);
        sum_loss += -(y * log(p) + (1 - y) * log(1 - p));
    }
    return sum_loss / probability.rows();
}

/**
 * @brief 给定预测值和真值标签，计算预测的精度Accuracy
 * 
 * @param probability 预测的概率值
 * @param labels      对应的真值
 * @return float 返回精度只，0-1
 */
float eval_test_accuracy(const Matrixf& probability, const Matrixu& labels){

    int success = 0;
    for(int i = 0; i < probability.rows(); ++i){
        auto row_ptr = probability.ptr(i);
        int predict_label = std::max_element(row_ptr, row_ptr + probability.cols()) - row_ptr;
        if(predict_label == labels.ptr(i)[0])
            success++;
    }
    return success / (float)probability.rows();
}

/**
 * @brief 返回一个数组，类似Python的range函数
 * 例如：range(3) = [0, 1, 2]
 * 
 * @param end 指定结束的位置，返回数组：(0, end]
 * @return vector<int> 
 */
vector<int> range(int end){
    vector<int> out(end);
    for(int i = 0; i < end; ++i)
        out[i] = i;
    return out;
}

/**
 * @brief 带动量的SGD优化器
 */
struct SGDMomentum{
    vector<Matrixf> delta_momentums;

    // 提供对应的参数params，和对应的梯度grads，进行参数的更新
    void update_params(const vector<Matrixf*>& params, const vector<Matrixf*>& grads, float lr, float momentum=0.9){

        if(delta_momentums.size() != params.size())
            delta_momentums.resize(params.size());

        for(int i =0 ; i < params.size(); ++i){
            auto& delta_momentum = delta_momentums[i];
            auto& param          = *params[i];
            auto& grad           = *grads[i];

            if(delta_momentum.numel() == 0)
                delta_momentum.resize(param.rows(), param.cols());
            
            delta_momentum = momentum * delta_momentum - lr * grad;
            param          = param + delta_momentum;
        }
    }
};

/**
 * @brief 保存模型到文件，序列化到文件
 * 
 * @param file   保存的文件名称
 * @param model  模型，实际上就是一堆矩阵
 */
bool save_model(const string& file, const vector<Matrixf>& model){

    ofstream out(file, ios::binary | ios::out);
    if(!out.is_open()){
        INFO("Open %s failed.", file.c_str());
        return false;
    }

    unsigned int header_file[] = {0x3355FF11, model.size()};
    out.write((char*)header_file, sizeof(header_file));

    for(auto& m : model){
        int header[] = {m.rows(), m.cols()};
        out.write((char*)header, sizeof(header));
        out.write((char*)m.ptr(), m.numel() * sizeof(float));
    }
    return out.good();
}

/**
 * @brief 从文件加载模型
 * 
 * @param file   模型文件，会做文件校验
 * @param model  模型数据，实际上就是一堆权重
 */
bool load_model(const string& file, vector<Matrixf>& model){

    ifstream in(file, ios::binary | ios::in);
    if(!in.is_open()){
        INFO("Open %s failed.", file.c_str());
        return false;
    }

    unsigned int header_file[2];
    in.read((char*)header_file, sizeof(header_file));

    if(header_file[0] != 0x3355FF11){
        INFO("Invalid model file: %s", file.c_str());
        return false;
    }

    model.resize(header_file[1]);
    for(int i = 0; i < model.size(); ++i){
        auto& m = model[i];
        int header[2];
        in.read((char*)header, sizeof(header));
        m.resize(header[0], header[1]);
        in.read((char*)m.ptr(), m.numel() * sizeof(float));
    }
    return in.good();
}

/**
 * @brief 执行测试集的测试，测试集有10000个图，进行识别并打印识别效果
 */
int do_test_dataset(){

    vector<Matrixf> model;
    if(!load_model("model.bin", model)){
        INFO("模型model.bin不存在，需要训练，do test dataset失败");
        return -1;
    }

    Matrixf input_to_hidden, hidden_bias, hidden_to_output, output_bias;
    input_to_hidden  = model[0];
    hidden_bias      = model[1];
    hidden_to_output = model[2];
    output_bias      = model[3];

    auto test_images      = load_mnist_images("t10k-images-idx3-ubyte");
    auto test_labels      = load_mnist_labels("t10k-labels-idx1-ubyte");
    auto test_norm_images = normalize_image_to_matrix(test_images);
    auto indexs           = range(test_images.rows());
    default_random_engine local_e(time(0));
    std::shuffle(indexs.begin(), indexs.end(), local_e);
    
    for(int i = 0; i < test_norm_images.rows(); ++i){
        int image_index  = indexs[i];
        auto item        = choice_rows(test_norm_images, indexs, i, 1);
        auto hidden      = (gemm_mul(item, false, input_to_hidden, false)    + hidden_bias).relu();
        auto probability = (gemm_mul(hidden, false, hidden_to_output, false) + output_bias).sigmoid();
        auto prob_ptr    = probability.ptr();
        int label        = std::max_element(prob_ptr, prob_ptr + probability.cols()) - prob_ptr;
        float confidence = prob_ptr[label];

        auto image       = Matrixu(28, 28, test_images.ptr(image_index));
        print_matrix(image);
        print_matrix(probability);
        INFO(
            color_text(
                "Predict is [%d], confidence is [%f]", 
                TextColor::Red
            ).c_str(), 
            label, confidence
        );
        INFO("=========================================================================");
        printf("Press 'Enter' to next, Press 'q' to quit: ");
        int c = getchar();
        if(c == 'q')
            break;
    }
    return 0;
}

/**
 * @brief 进行bmp文件检测测试
 * 要求bmp文件必须是28x28的24bit文件
 */
int do_test_bmp(const char* file){

    vector<Matrixf> model;
    if(!load_model("model.bin", model)){
        INFO("模型model.bin不存在，需要训练，do test bmp失败");
        return -1;
    }

    Matrixf input_to_hidden, hidden_bias, hidden_to_output, output_bias;
    input_to_hidden  = model[0];
    hidden_bias      = model[1];
    hidden_to_output = model[2];
    output_bias      = model[3];

    auto image       = load_bmp_matrix(file);
    if(image.empty()){
        INFO("Load BMP %s failed", file);
        return -1;
    }

    auto item        = normalize_image_to_matrix(image);
    auto hidden      = (gemm_mul(item, false, input_to_hidden, false)    + hidden_bias).relu();
    auto probability = (gemm_mul(hidden, false, hidden_to_output, false) + output_bias).sigmoid();
    auto prob_ptr    = probability.ptr();
    int label        = std::max_element(prob_ptr, prob_ptr + probability.cols()) - prob_ptr;
    float confidence = prob_ptr[label];

    print_matrix(Matrixu(28, 28, image.ptr()));
    print_matrix(probability);
    INFO(
        color_text(
            "BMP %s Predict is [%d], confidence is [%f]",
            TextColor::Red
        ).c_str(), 
    file, label, confidence);
    INFO("=========================================================================");
    return 0;
}

/**
 * @brief 进行训练任务，训练结束后模型储存为model.bin
 */
int do_train(){

    auto train_images        = load_mnist_images("train-images-idx3-ubyte");
    auto train_labels        = load_mnist_labels("train-labels-idx1-ubyte");
    auto train_norm_images   = normalize_image_to_matrix(train_images);
    auto train_onehot_labels = label_to_onehot(train_labels);

    auto test_images         = load_mnist_images("t10k-images-idx3-ubyte");
    auto test_labels         = load_mnist_labels("t10k-labels-idx1-ubyte");
    auto test_norm_images    = normalize_image_to_matrix(test_images);
    auto test_onehot_labels  = label_to_onehot(test_labels);

    int num_images  = train_norm_images.rows();
    int num_input   = train_norm_images.cols();
    int num_hidden  = 1024;
    int num_output  = 10;
    int num_epoch   = 10;
    float lr        = 1e-1;
    int batch_size  = 256;
    float momentum  = 0.9f;
    int num_batch_per_epoch = num_images / batch_size;
    auto image_indexs       = range(num_images);

    // 凯明初始化，fan_in + fan_out
    Matrixf input_to_hidden  = create_normal_distribution_matrix(num_input,  num_hidden, 0, 2.0f / sqrt((float)(num_input + num_hidden)));
    Matrixf hidden_bias(1, num_hidden);
    Matrixf hidden_to_output = create_normal_distribution_matrix(num_hidden, num_output, 0, 1.0f / sqrt((float)(num_hidden + num_output)));
    Matrixf output_bias(1, num_output);
    SGDMomentum optim;
    for(int epoch = 0; epoch < num_epoch; ++epoch){

        if(epoch == 8){
            lr *= 0.1;
        }

        // 打乱索引
        std::shuffle(image_indexs.begin(), image_indexs.end(), global_random_engine);
        
        // 开始循环所有的batch
        for(int ibatch = 0; ibatch < num_batch_per_epoch; ++ibatch){

            // 前向过程
            auto x           = choice_rows(train_norm_images,   image_indexs, ibatch * batch_size, batch_size);
            auto y           = choice_rows(train_onehot_labels, image_indexs, ibatch * batch_size, batch_size);
            auto hidden      = gemm_mul(x,          false, input_to_hidden,  false) + hidden_bias;
            auto hidden_act  = hidden.relu();
            auto output      = gemm_mul(hidden_act, false, hidden_to_output, false) + output_bias;
            auto probability = output.sigmoid();
            float loss       = compute_loss(probability, y);

            if(ibatch % 50 == 0){
                INFO("Epoch %.2f / %d, Loss: %f, LR: %f", epoch + ibatch / (float)num_batch_per_epoch, num_epoch, loss, lr);
            }

            // 反向过程
            // C = AB
            // dA = G * BT
            // dB = AT * G
            // loss部分求导，loss对output求导
            auto doutput           = (probability - y) / batch_size;

            // 第二个Linear求导
            auto doutput_bias      = row_sum(doutput);
            auto dhidden_to_output = gemm_mul(hidden_act, true, doutput, false);
            auto dhidden_act       = gemm_mul(doutput, false, hidden_to_output, true);

            // 第一个Linear输出求导
            auto dhidden           = delta_relu(dhidden_act, hidden);

            // 第一个Linear求导
            auto dinput_to_hidden  = gemm_mul(x, true, dhidden, false);
            auto dhidden_bias      = row_sum(dhidden);

            // 调用优化器来调整更新参数
            optim.update_params(
                {&input_to_hidden,  &hidden_bias,  &hidden_to_output,  &output_bias},
                {&dinput_to_hidden, &dhidden_bias, &dhidden_to_output, &doutput_bias},
                lr, momentum
            );
        }

        // 模型对测试集进行测试，并打印精度
        auto test_hidden      = (gemm_mul(test_norm_images, input_to_hidden) + hidden_bias).relu();
        auto test_probability = (gemm_mul(test_hidden, hidden_to_output)     + output_bias).sigmoid();
        float accuracy        = eval_test_accuracy(test_probability, test_labels);
        float test_loss       = compute_loss(test_probability, test_onehot_labels);
        INFO("Test Accuracy: %.2f %%, Loss: %f", accuracy * 100, test_loss);
    }

    INFO("Save model to model.bin");
    save_model("model.bin", {input_to_hidden, hidden_bias, hidden_to_output, output_bias});
    return 0;
}

int main(int argc, char** argv){

    const char* method = "";
    if(argc > 1){
        method = argv[1];
    }

    if(strcmp(method, "image") == 0){
        const char* file = "5.bmp";
        if(argc > 2)
            file = argv[2];
        
        return do_test_bmp(file);
    }

    if(strcmp(method, "train") == 0)
        return do_train();

    if(strcmp(method, "test") == 0)
        return do_test_dataset();

    printf(
        "Help: \n"
        "     ./pro train        执行训练\n"
        "     ./pro test         执行测试\n"
        "     ./pro image 5.bmp  加载28x28的bmp图像文件进行预测\n"
    );
    return 0;
}