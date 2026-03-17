# 说明：导入标准库中的数学函数模块，用于 sigmoid 和指数运算。
import math
# 说明：导入标准库中的随机数模块，用于生成模拟数据和打乱样本。
import random

# 说明：设置全局随机种子，保证每次运行生成的数据和训练过程可复现。
random.seed(42)

# 说明：定义 sigmoid 函数，输入是一个标量 z（shape: 标量），输出是概率（shape: 标量）。
def sigmoid(z):
    # 说明：返回 1 / (1 + exp(-z))，用于二分类概率建模。
    return 1.0 / (1.0 + math.exp(-z))

# 说明：定义二维点与矩阵相乘的函数，x shape=(2,)，W shape=(h,2)，输出 h shape=(h,)。
def linear_forward_2_to_h(x, W, b):
    # 说明：创建输出向量 h，长度为隐藏维度 h_dim，初始值为 0。
    h = [0.0 for _ in range(len(W))]
    # 说明：遍历每个隐藏单元 i（对应 W 的第 i 行）。
    for i in range(len(W)):
        # 说明：线性变换 h[i] = W[i][0]*x[0] + W[i][1]*x[1] + b[i]。
        h[i] = W[i][0] * x[0] + W[i][1] * x[1] + b[i]
    # 说明：返回隐藏表示 h，shape=(h_dim,)。
    return h

# 说明：定义隐藏向量与权重的点积函数，h shape=(h_dim,)，w shape=(h_dim,)，输出标量。
def linear_forward_h_to_1(h, w, b):
    # 说明：初始化标量输出 z，起始值是偏置 b。
    z = b
    # 说明：累加每个维度的乘积 w[j]*h[j]。
    for j in range(len(h)):
        # 说明：将当前维度的线性贡献加入 z。
        z += w[j] * h[j]
    # 说明：返回线性输出 z（shape: 标量）。
    return z

# 说明：生成源域和目标域的二维二分类数据。
def make_domain_data(n_per_class, shift_x):
    # 说明：初始化样本列表，每个元素是 (x, y)，x shape=(2,)，y 是 0/1。
    data = []
    # 说明：循环生成类别 0 的样本，类别 0 以 (-1+shift_x, -1) 为中心。
    for _ in range(n_per_class):
        # 说明：x0 的第 1 维来自均值附近的随机扰动。
        x0 = random.gauss(-1.0 + shift_x, 0.6)
        # 说明：x1 的第 2 维来自均值附近的随机扰动。
        x1 = random.gauss(-1.0, 0.6)
        # 说明：将类别 0 的样本加入列表，x shape=(2,)，y=0。
        data.append(([x0, x1], 0))
    # 说明：循环生成类别 1 的样本，类别 1 以 (1+shift_x, 1) 为中心。
    for _ in range(n_per_class):
        # 说明：x0 的第 1 维来自均值附近的随机扰动。
        x0 = random.gauss(1.0 + shift_x, 0.6)
        # 说明：x1 的第 2 维来自均值附近的随机扰动。
        x1 = random.gauss(1.0, 0.6)
        # 说明：将类别 1 的样本加入列表，x shape=(2,)，y=1。
        data.append(([x0, x1], 1))
    # 说明：返回该域的数据列表，长度是 2*n_per_class。
    return data

# 说明：计算在给定数据上的标签分类准确率。
def evaluate_label_accuracy(data, Wf, bf, wy, by):
    # 说明：初始化预测正确计数器。
    correct = 0
    # 说明：遍历每个样本 (x, y_true)。
    for x, y_true in data:
        # 说明：前向计算隐藏特征 h，shape=(h_dim,)。
        h = linear_forward_2_to_h(x, Wf, bf)
        # 说明：前向计算标签 logits，shape=标量。
        z_y = linear_forward_h_to_1(h, wy, by)
        # 说明：将概率 >=0.5 判为 1，否则为 0。
        y_pred = 1 if sigmoid(z_y) >= 0.5 else 0
        # 说明：若预测与真实一致，则计数加 1。
        if y_pred == y_true:
            correct += 1
    # 说明：返回准确率 = 正确数 / 总样本数。
    return correct / float(len(data))

# 说明：脚本主入口，执行数据构造、训练和评估。
def main():
    # 说明：设置输入维度 in_dim=2（二维点）。
    in_dim = 2
    # 说明：设置隐藏维度 h_dim=4（特征提取器输出维度）。
    h_dim = 4
    # 说明：设置训练轮数 epochs。
    epochs = 60
    # 说明：设置学习率 lr。
    lr = 0.02
    # 说明：设置对抗强度 lambda_adv，用于梯度反转。
    lambda_adv = 0.4

    # 说明：生成源域训练数据，域偏移 shift_x=0.0。
    source_train = make_domain_data(n_per_class=200, shift_x=0.0)
    # 说明：生成目标域训练数据（无标签参与对抗），域偏移 shift_x=1.2。
    target_train = make_domain_data(n_per_class=200, shift_x=1.2)
    # 说明：生成目标域测试数据（用于最终分类评估），同样偏移 1.2。
    target_test = make_domain_data(n_per_class=200, shift_x=1.2)

    # 说明：初始化特征提取器参数 Wf，shape=(h_dim,2)。
    Wf = [[random.uniform(-0.2, 0.2) for _ in range(in_dim)] for _ in range(h_dim)]
    # 说明：初始化特征提取器偏置 bf，shape=(h_dim,)。
    bf = [0.0 for _ in range(h_dim)]
    # 说明：初始化标签分类器权重 wy，shape=(h_dim,)。
    wy = [random.uniform(-0.2, 0.2) for _ in range(h_dim)]
    # 说明：初始化标签分类器偏置 by，shape=标量。
    by = 0.0
    # 说明：初始化域分类器权重 wd，shape=(h_dim,)。
    wd = [random.uniform(-0.2, 0.2) for _ in range(h_dim)]
    # 说明：初始化域分类器偏置 bd，shape=标量。
    bd = 0.0

    # 说明：打印训练开始信息。
    print("开始训练 Domain Adversarial (简化 DANN) 模型...")

    # 说明：按 epoch 进行迭代训练。
    for epoch in range(1, epochs + 1):
        # 说明：构造批次样本列表，每个元素是 (x, y_label_or_none, d_domain)。
        mix = []
        # 说明：加入源域样本，域标签 d=0，且有类别标签 y。
        for x, y in source_train:
            # 说明：把源域样本打包进混合列表。
            mix.append((x, y, 0))
        # 说明：加入目标域样本，域标签 d=1，但标签任务中 y 不使用。
        for x, _ in target_train:
            # 说明：目标域 y 用 None 占位，表示无监督标签。
            mix.append((x, None, 1))
        # 说明：随机打乱混合样本，减少训练顺序偏置。
        random.shuffle(mix)

        # 说明：遍历每个样本执行一次 SGD 更新。
        for x, y, d in mix:
            # 说明：前向得到隐藏特征 h，shape=(h_dim,)。
            h = linear_forward_2_to_h(x, Wf, bf)

            # 说明：前向得到域分类器输出概率 p_d，shape=标量。
            z_d = linear_forward_h_to_1(h, wd, bd)
            # 说明：将域 logits 通过 sigmoid 转为概率。
            p_d = sigmoid(z_d)
            # 说明：二元交叉熵对 logits 的梯度是 (p_d - d)。
            g_zd = p_d - float(d)

            # 说明：计算域分支对隐藏特征的梯度 g_h_domain，shape=(h_dim,)。
            g_h_domain = [g_zd * wd[j] for j in range(h_dim)]

            # 说明：初始化标签分支对隐藏特征的梯度 g_h_label，默认全 0。
            g_h_label = [0.0 for _ in range(h_dim)]

            # 说明：若样本来自源域（有标签），则计算标签分类损失梯度。
            if y is not None:
                # 说明：前向得到标签分类 logits z_y，shape=标量。
                z_y = linear_forward_h_to_1(h, wy, by)
                # 说明：标签分类概率 p_y，shape=标量。
                p_y = sigmoid(z_y)
                # 说明：标签损失对 logits 的梯度 g_zy = p_y - y。
                g_zy = p_y - float(y)
                # 说明：更新标签分类器权重 wy（shape=(h_dim,)）。
                for j in range(h_dim):
                    # 说明：SGD: wy[j] -= lr * g_zy * h[j]。
                    wy[j] -= lr * g_zy * h[j]
                # 说明：更新标签分类器偏置 by（标量）。
                by -= lr * g_zy
                # 说明：计算标签分支对隐藏特征的梯度 g_h_label。
                for j in range(h_dim):
                    # 说明：g_h_label[j] = g_zy * wy[j]。
                    g_h_label[j] = g_zy * wy[j]

            # 说明：先更新域分类器参数（不反转，因为它本身要区分域）。
            for j in range(h_dim):
                # 说明：wd[j] -= lr * g_zd * h[j]。
                wd[j] -= lr * g_zd * h[j]
            # 说明：更新域分类器偏置 bd。
            bd -= lr * g_zd

            # 说明：组合传给特征提取器的梯度：标签梯度 - lambda*域梯度（梯度反转）。
            g_h_total = [g_h_label[j] - lambda_adv * g_h_domain[j] for j in range(h_dim)]

            # 说明：根据 g_h_total 反向更新特征提取器 Wf 和 bf。
            for j in range(h_dim):
                # 说明：Wf[j][0] 的梯度是 g_h_total[j] * x[0]。
                Wf[j][0] -= lr * g_h_total[j] * x[0]
                # 说明：Wf[j][1] 的梯度是 g_h_total[j] * x[1]。
                Wf[j][1] -= lr * g_h_total[j] * x[1]
                # 说明：bf[j] 的梯度是 g_h_total[j]。
                bf[j] -= lr * g_h_total[j]

        # 说明：每 10 个 epoch 打印一次目标域准确率。
        if epoch % 10 == 0:
            # 说明：在目标测试集上评估标签分类准确率。
            acc_t = evaluate_label_accuracy(target_test, Wf, bf, wy, by)
            # 说明：打印当前 epoch 和目标域准确率。
            print(f"Epoch {epoch:02d} | Target Acc = {acc_t:.4f}")

    # 说明：计算并打印最终源域和目标域准确率，方便观察泛化情况。
    source_acc = evaluate_label_accuracy(source_train, Wf, bf, wy, by)
    # 说明：计算目标域最终准确率。
    target_acc = evaluate_label_accuracy(target_test, Wf, bf, wy, by)
    # 说明：输出最终结果。
    print("训练结束。")
    # 说明：打印源域训练准确率。
    print(f"Final Source Acc: {source_acc:.4f}")
    # 说明：打印目标域测试准确率。
    print(f"Final Target Acc: {target_acc:.4f}")

# 说明：当此文件被直接运行时，执行 main 函数。
if __name__ == "__main__":
    # 说明：调用主函数启动流程。
    main()
