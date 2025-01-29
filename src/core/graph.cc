#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        if (!this->topo_sort()) {
            std::cout << "!! topo_sort failed" << std::endl;
            return;
        }
        
        for (int i = 0; i < (int)ops.size(); i ++) {
            auto now_op = ops[i];
            // std::cout << "Processing Operation: " << now_op->toString() << std::endl;

            switch (now_op->getOpType().underlying()) {
            default: {
                break;
            }
            case OpType::Transpose : { // 去除冗余的算子
                auto tensor0 = now_op->getInputs(0);
                auto tensor1 = now_op->getOutputs()[0];
                auto next_op = tensor1->getTargets()[0];
                if (next_op && next_op->getOpType() == OpType::Transpose &&
                    tensor1 == next_op->getInputs(0))
                {
                    auto tensor3 = next_op->getOutputs()[0];

                    // tensor3->setSource(nullptr);
                    tensor0->removeTarget(now_op);  // important
                    for (auto &targetOp  : tensor3->getTargets()) {
                        targetOp->replaceInput(tensor3, tensor0);
                        tensor0->addTarget(targetOp);
                        targetOp->removePredecessors(next_op);
                    }
                    for (auto pred : now_op->getPredecessors()) {
                        pred->removeSuccessors(now_op);
                    }

                    removeTensor(tensor1);
                    removeTensor(tensor3);
                    // ops.erase(ops.begin() + i, ops.begin() + i + 2);
                    removeOperator(now_op);
                    removeOperator(next_op);
                    i --;
                }
                break;
            }
			case OpType::MatMul: { // 合并算子
                auto matmulOp = as<MatmulObj>(now_op);
                // bool optimized = false; // 用于标记当前 MatMul 是否优化过
                for (int cc = 0; cc < 2; cc ++) {
                    auto matmulInput = matmulOp->getInputs()[cc];

                    // 检查输入是否为 Transpose
                    if (matmulInput) { 
                        auto inputOp = matmulInput->getSource();
                        if (inputOp && inputOp->getOpType() == OpType::Transpose) {
                            auto transposeOp = as<TransposeObj>(inputOp);
                            std::cout << "-current input " << (cc+1) << " is " << matmulInput->toString() << std::endl;
                            std::cout << "-current input " << (cc+1) << " is a Transpose operation: " << transposeOp->toString() << std::endl;

                            // 判断 Transpose 是否对最后两个维度的交换
                            auto permute = transposeOp->getPermute();
                            if (permute.size() >= 2 &&
                                permute[permute.size() - 2] == static_cast<int>(permute.size() - 1) &&
                                permute[permute.size() - 1] == static_cast<int>(permute.size() - 2)) 
                            {
                                std::cout << "Found Transpose on Input that swaps the last two dimensions." << std::endl;

                                // 更新 MatMul 的 transA/B 属性
                                if (cc == 0)
                                    matmulOp->setTransA(!matmulOp->getTransA());
                                else
                                    matmulOp->setTransB(!matmulOp->getTransB());

                                // if (!transposeOp->getOutputs().empty()) {
                                // std::cout << "Changing MatMul Input from Transpose->Input to Transpose's Source." << std::endl;
                                auto transposeOutput = transposeOp->getOutputs()[0]; // 获取 Transpose 的输出张量
                                auto transposeInput = transposeOp->getInputs(0); // 获取 Transpose 的输入张量

                                transposeInput->addTarget(matmulOp);

                                // 删除 Transpose 输出张量的来源连接
                                transposeOutput->setSource(nullptr);

                                // 将 Transpose 从其输入张量的 targets 列表中移除
                                matmulInput->removeTarget(transposeOp);
                                matmulOp->replaceInput(matmulInput, transposeInput);

                                removeOperator(transposeOp);
                                removeTensor(matmulInput);

                                this->print();
                            }
                        }
                    }
                }
                break;
            }
            }
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        
        std::vector<size_t> tensor_offset_vec = std::vector<size_t>(tensors.size());
        for (size_t i = 0; i < tensors.size(); i++)
        {
            auto offset = allocator.alloc(tensors[i]->getBytes());
            tensor_offset_vec[i] = offset;
        }
        auto start_ptr= allocator.getPtr();
        for (size_t i = 0; i < tensors.size(); i++)
        {
            auto offset = tensor_offset_vec[i];
            // 指针加上偏移量
            void *ptr = reinterpret_cast<char *>(start_ptr) + offset;
            // new blob
            tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, ptr));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini