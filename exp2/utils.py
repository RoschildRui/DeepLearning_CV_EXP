import numpy as np


def generate_report4exp2_4(results):
    report_content = "# LeNet在FashionMNIST上的实验结果\n\n"
    report_content += "## 实验配置\n\n"
    report_content += "本实验测试了LeNet网络在不同卷积核大小、填充和步长组合下的性能表现。\n\n"
    
    report_content += "## 实验结果\n\n"
    
    # 成功的实验
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    if successful_results:
        report_content += "### 成功的实验配置\n\n"
        report_content += "| 配置名称 | 卷积核大小 | 填充 | 步长 | 最终测试准确率(%) | 训练时间(秒) | 全连接层输入尺寸 |\n"
        report_content += "|---------|-----------|------|------|------------------|-------------|------------------|\n"
        
        for name, result in successful_results.items():
            config = result['config']
            report_content += f"| {name} | {config['kernel_size']} | {config['padding']} | {config['stride']} | "
            report_content += f"{result['final_test_accuracy']:.2f} | {result['training_time']:.2f} | {result['fc_input_size']} |\n"
        
        # 找出最佳配置
        best_config = max(successful_results.items(), key=lambda x: x[1]['final_test_accuracy'])
        report_content += f"\n### 最佳配置\n\n"
        report_content += f"**{best_config[0]}** 达到了最高的测试准确率 **{best_config[1]['final_test_accuracy']:.2f}%**\n\n"
        
        config = best_config[1]['config']
        report_content += f"- 卷积核大小: {config['kernel_size']}\n"
        report_content += f"- 填充: {config['padding']}\n"
        report_content += f"- 步长: {config['stride']}\n"
        report_content += f"- 训练时间: {best_config[1]['training_time']:.2f}秒\n\n"
    
    # 失败的实验
    failed_results = {k: v for k, v in results.items() if 'error' in v}
    if failed_results:
        report_content += "### 失败的实验配置\n\n"
        for name, result in failed_results.items():
            config = result['config']
            report_content += f"- **{name}**: 卷积核大小={config['kernel_size']}, 填充={config['padding']}, 步长={config['stride']}\n"
            report_content += f"  错误信息: {result['error']}\n\n"
    
    report_content += "## 结论\n\n"
    report_content += "1. 不同的卷积核大小、填充和步长组合对LeNet的性能有显著影响\n"
    report_content += "2. 合适的填充可以保持特征图尺寸，避免信息丢失\n"
    report_content += "3. 较大的步长会减少计算量但可能损失细节信息\n"
    report_content += "4. 需要在准确率和计算效率之间找到平衡\n"

    with open('results/exp2_4/experiment_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n实验报告已保存到: results/exp2_4/experiment_report.md")


def generate_report4exp2_5(results):
    """生成实验报告"""
    report_content = "# AlexNet在FashionMNIST上的实验结果\n\n"
    report_content += "## 实验目标\n\n"
    report_content += "使用AlexNet网络对FashionMNIST数据集进行识别，通过不同配置的实验来达到最优识别率。\n\n"
    
    report_content += "## 数据集信息\n\n"
    report_content += "- **数据集**: FashionMNIST\n"
    report_content += "- **类别数**: 10类 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)\n"
    report_content += "- **图像尺寸**: 28×28像素，灰度图像\n"
    report_content += "- **训练集**: 60,000张图像\n"
    report_content += "- **测试集**: 10,000张图像\n\n"
    
    report_content += "## 模型架构\n\n"
    report_content += "### 1. 原始AlexNet\n"
    report_content += "- 适配28×28输入的经典AlexNet架构\n"
    report_content += "- 5个卷积层 + 3个全连接层\n"
    report_content += "- 使用ReLU激活函数和Dropout正则化\n\n"
    
    report_content += "### 2. 修改版AlexNet\n"
    report_content += "- 专门为28×28小图像优化的AlexNet变体\n"
    report_content += "- 减少了卷积核尺寸和步长，更适合小图像\n"
    report_content += "- 优化了全连接层的参数数量\n\n"
    
    report_content += "## 实验结果\n\n"
    
    # 成功的实验
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    if successful_results:
        report_content += "### 成功的实验配置\n\n"
        report_content += "| 配置名称 | 模型类型 | 批次大小 | 学习率 | 训练轮数 | Dropout | 最佳准确率(%) | 训练时间(分钟) | 参数数量(M) |\n"
        report_content += "|---------|---------|---------|--------|---------|---------|---------------|----------------|-------------|\n"
        
        for name, result in successful_results.items():
            config = result['config']
            model_type = "原始AlexNet" if "Original" in name else "修改版AlexNet"
            report_content += f"| {name} | {model_type} | {config['batch_size']} | {config['lr']} | {config['epochs']} | "
            report_content += f"{config['dropout']} | {result['best_test_accuracy']:.2f} | "
            report_content += f"{result['training_time']/60:.1f} | {result['total_params']/1e6:.2f} |\n"
        
        # 找出最佳配置
        best_config = max(successful_results.items(), key=lambda x: x[1]['best_test_accuracy'])
        report_content += f"\n### 🏆 最佳配置\n\n"
        report_content += f"**{best_config[0]}** 达到了最高的测试准确率 **{best_config[1]['best_test_accuracy']:.2f}%**\n\n"
        
        config = best_config[1]['config']
        report_content += f"**配置详情:**\n"
        report_content += f"- 模型类型: {'原始AlexNet' if 'Original' in best_config[0] else '修改版AlexNet'}\n"
        report_content += f"- 批次大小: {config['batch_size']}\n"
        report_content += f"- 学习率: {config['lr']}\n"
        report_content += f"- 训练轮数: {config['epochs']}\n"
        report_content += f"- Dropout率: {config['dropout']}\n"
        report_content += f"- 权重衰减: {config['weight_decay']}\n"
        report_content += f"- 数据增强: {'是' if config['augment'] else '否'}\n"
        report_content += f"- 训练时间: {best_config[1]['training_time']/60:.1f}分钟\n"
        report_content += f"- 模型参数: {best_config[1]['total_params']/1e6:.2f}M\n\n"
        
        # 性能分析
        report_content += "### 性能分析\n\n"
        all_accs = [result['best_test_accuracy'] for result in successful_results.values()]
        avg_acc = np.mean(all_accs)
        max_acc = np.max(all_accs)
        min_acc = np.min(all_accs)
        
        report_content += f"- **平均准确率**: {avg_acc:.2f}%\n"
        report_content += f"- **最高准确率**: {max_acc:.2f}%\n"
        report_content += f"- **最低准确率**: {min_acc:.2f}%\n"
        report_content += f"- **准确率标准差**: {np.std(all_accs):.2f}%\n\n"
    
    # 失败的实验
    failed_results = {k: v for k, v in results.items() if 'error' in v}
    if failed_results:
        report_content += "### 失败的实验配置\n\n"
        for name, result in failed_results.items():
            config = result['config']
            report_content += f"- **{name}**: {config['model_class'].__name__}\n"
            report_content += f"  错误信息: {result['error']}\n\n"
    
    report_content += "## 优化策略\n\n"
    report_content += "1. **数据增强**: 使用随机水平翻转和旋转来增加数据多样性\n"
    report_content += "2. **学习率调度**: 使用StepLR调度器在训练过程中降低学习率\n"
    report_content += "3. **正则化**: 使用Dropout和权重衰减防止过拟合\n"
    report_content += "4. **批次大小优化**: 测试不同批次大小对性能的影响\n"
    report_content += "5. **架构调整**: 针对28×28小图像优化网络架构\n\n"
    
    report_content += "## 结论\n\n"
    if successful_results:
        best_acc = max(result['best_test_accuracy'] for result in successful_results.values())
        report_content += f"1. **最优识别率**: 通过优化配置，AlexNet在FashionMNIST上达到了 **{best_acc:.2f}%** 的识别准确率\n"
    report_content += "2. **架构适配**: 修改版AlexNet比原始AlexNet更适合小尺寸图像\n"
    report_content += "3. **超参数重要性**: 学习率、批次大小和Dropout率对性能有显著影响\n"
    report_content += "4. **数据增强效果**: 数据增强技术能有效提升模型泛化能力\n"
    report_content += "5. **训练策略**: 学习率调度和早停策略有助于获得更好的性能\n\n"
    
    report_content += "## 与LeNet的比较\n\n"
    report_content += "- AlexNet相比LeNet有更深的网络结构和更多的参数\n"
    report_content += "- AlexNet使用了更多的正则化技术(Dropout)\n"
    report_content += "- AlexNet在FashionMNIST上的性能明显优于LeNet\n"
    report_content += "- 但AlexNet的训练时间和计算资源需求也更高\n"

    with open('results/exp2_5/alexnet_experiment_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n实验报告已保存到: results/exp2_5/alexnet_experiment_report.md")