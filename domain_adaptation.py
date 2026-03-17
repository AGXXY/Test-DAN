# 说明：导入数学模块，用于指数函数和开方计算。
import math
# 说明：导入随机模块，用于生成可复现的模拟数据。
import random

# 说明：固定随机种子，保证每次运行结果稳定。
random.seed(7)

# 说明：定义 sigmoid 函数，输入标量 logits，输出区间 (0,1) 的概率。
def sigmoid(z):
    # 说明：返回标准 sigmoid 计算结果。
    return 1.0 / (1.0 + math.exp(-z))

# 说明：生成二维二分类数据；shift_x 用于制造域偏移。
def make_domain_data(n_per_class, shift_x):
    # 说明：初始化数据列表，每个元素是 (x, y)，x shape=(2,)，y 是标量 0/1。
    data = []
    # 说明：生成类别 0 样本，中心约为 (-1+shift_x, -1)。
    for _ in range(n_per_class):
        # 说明：生成第 1 维特征。
        x0 = random.gauss(-1.0 + shift_x, 0.7)
        # 说明：生成第 2 维特征。
        x1 = random.gauss(-1.0, 0.7)
        # 说明：追加一个类别 0 样本。
        data.append(([x0, x1], 0))
    # 说明：生成类别 1 样本，中心约为 (1+shift_x, 1)。
    for _ in range(n_per_class):
        # 说明：生成第 1 维特征。
        x0 = random.gauss(1.0 + shift_x, 0.7)
        # 说明：生成第 2 维特征。
        x1 = random.gauss(1.0, 0.7)
        # 说明：追加一个类别 1 样本。
        data.append(([x0, x1], 1))
    # 说明：返回完整数据集。
    return data

# 说明：计算一个域数据的逐维均值和标准差。
def mean_std_of_x(data):
    # 说明：初始化总和向量 sum_x，shape=(2,)。
    sum_x = [0.0, 0.0]
    # 说明：遍历所有样本以累加两维特征。
    for x, _ in data:
        # 说明：累加第 1 维。
        sum_x[0] += x[0]
        # 说明：累加第 2 维。
        sum_x[1] += x[1]
    # 说明：样本总数 n（标量）。
    n = float(len(data))
    # 说明：计算均值 mu，shape=(2,)。
    mu = [sum_x[0] / n, sum_x[1] / n]

    # 说明：初始化方差累计向量 var，shape=(2,)。
    var = [0.0, 0.0]
    # 说明：再次遍历样本计算每一维方差。
    for x, _ in data:
        # 说明：累加第 1 维平方差。
        var[0] += (x[0] - mu[0]) * (x[0] - mu[0])
        # 说明：累加第 2 维平方差。
        var[1] += (x[1] - mu[1]) * (x[1] - mu[1])
    # 说明：计算标准差 std，增加极小值避免除零，shape=(2,)。
    std = [math.sqrt(var[0] / n) + 1e-8, math.sqrt(var[1] / n) + 1e-8]
    # 说明：返回均值和标准差。
    return mu, std

# 说明：把源域特征做“均值+方差”对齐到目标域统计量。
def adapt_source_to_target_stats(source_data, mu_s, std_s, mu_t, std_t):
    # 说明：初始化对齐后的数据列表，元素仍是 (x_new, y)。
    out = []
    # 说明：逐样本执行特征变换。
    for x, y in source_data:
        # 说明：对第 1 维执行 (x-mu_s)/std_s*std_t + mu_t。
        x0_new = ((x[0] - mu_s[0]) / std_s[0]) * std_t[0] + mu_t[0]
        # 说明：对第 2 维执行同样变换。
        x1_new = ((x[1] - mu_s[1]) / std_s[1]) * std_t[1] + mu_t[1]
        # 说明：将变换后的样本和原标签写入输出。
        out.append(([x0_new, x1_new], y))
    # 说明：返回对齐后的源域数据（位于目标统计空间）。
    return out

# 说明：训练一个二维逻辑回归分类器（纯 Python SGD 实现）。
def train_logistic(data, epochs, lr):
    # 说明：初始化权重 w，shape=(2,)。
    w = [0.0, 0.0]
    # 说明：初始化偏置 b（标量）。
    b = 0.0
    # 说明：按 epoch 迭代训练。
    for _ in range(epochs):
        # 说明：每轮先打乱样本，提升 SGD 效果。
        random.shuffle(data)
        # 说明：逐样本更新参数。
        for x, y in data:
            # 说明：线性输出 z = w0*x0 + w1*x1 + b。
            z = w[0] * x[0] + w[1] * x[1] + b
            # 说明：概率预测 p，shape=标量。
            p = sigmoid(z)
            # 说明：交叉熵对 z 的梯度 g = p - y。
            g = p - float(y)
            # 说明：更新权重第 1 维。
            w[0] -= lr * g * x[0]
            # 说明：更新权重第 2 维。
            w[1] -= lr * g * x[1]
            # 说明：更新偏置。
            b -= lr * g
    # 说明：返回训练好的参数。
    return w, b

# 说明：计算分类准确率。
def evaluate(data, w, b):
    # 说明：初始化正确样本计数。
    correct = 0
    # 说明：遍历每个样本。
    for x, y in data:
        # 说明：计算 logits。
        z = w[0] * x[0] + w[1] * x[1] + b
        # 说明：阈值 0.5 进行二分类。
        y_hat = 1 if sigmoid(z) >= 0.5 else 0
        # 说明：统计正确预测数。
        if y_hat == y:
            correct += 1
    # 说明：返回准确率。
    return correct / float(len(data))

# 说明：脚本主流程，演示“无自适应 vs 统计对齐自适应”。
def main():
    # 说明：构建源域训练集（shift_x=0.0）。
    source_train = make_domain_data(n_per_class=250, shift_x=0.0)
    # 说明：构建目标域无标签数据（用于估计目标统计量）。
    target_unlabeled = make_domain_data(n_per_class=250, shift_x=1.1)
    # 说明：构建目标域测试集（用于评估迁移效果）。
    target_test = make_domain_data(n_per_class=250, shift_x=1.1)

    # 说明：训练 baseline（只用原始源域数据）。
    w_base, b_base = train_logistic(data=source_train[:], epochs=35, lr=0.04)
    # 说明：计算 baseline 在目标域上的准确率。
    acc_base = evaluate(target_test, w_base, b_base)

    # 说明：计算源域统计量：mu_s/std_s，shape 都是 (2,)。
    mu_s, std_s = mean_std_of_x(source_train)
    # 说明：计算目标域统计量：mu_t/std_t，shape 都是 (2,)。
    mu_t, std_t = mean_std_of_x(target_unlabeled)

    # 说明：把源域训练数据对齐到目标统计空间。
    source_aligned = adapt_source_to_target_stats(source_train, mu_s, std_s, mu_t, std_t)
    # 说明：在对齐后的源域数据上重新训练分类器。
    w_adapt, b_adapt = train_logistic(data=source_aligned, epochs=35, lr=0.04)
    # 说明：在目标测试集上评估自适应模型。
    acc_adapt = evaluate(target_test, w_adapt, b_adapt)

    # 说明：打印结果标题。
    print("Domain Adaptation (统计对齐) 结果：")
    # 说明：打印 baseline 结果。
    print(f"Baseline Target Acc: {acc_base:.4f}")
    # 说明：打印 adaptation 结果。
    print(f"Adapted  Target Acc: {acc_adapt:.4f}")

    # 说明：额外打印变量形状与含义，方便你阅读和二次开发。
    print("\n变量形状说明：")
    # 说明：说明单个输入样本 x 的形状。
    print("x: shape=(2,), 含义=[feature_1, feature_2]")
    # 说明：说明逻辑回归权重 w 的形状。
    print("w: shape=(2,), 含义=分类器对两个特征的线性权重")
    # 说明：说明偏置 b 的形状。
    print("b: shape=(), 含义=分类器偏置")
    # 说明：说明统计量 mu/std 的形状。
    print("mu_s/std_s/mu_t/std_t: shape=(2,), 含义=源/目标域逐维均值和标准差")

# 说明：当脚本直接运行时，执行 main。
if __name__ == "__main__":
    # 说明：调用主函数。
    main()
