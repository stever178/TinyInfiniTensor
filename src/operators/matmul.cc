#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        IT_ASSERT(shapeA.size() == shapeB.size());

        if (this->getTransA())
        {
            std::swap(shapeA[shapeA.size() - 1], shapeA[shapeA.size() - 2]);
        }

        if (this->getTransB())
        {
            std::swap(shapeB[shapeB.size() - 1], shapeB[shapeB.size() - 2]);
        }

        IT_ASSERT(shapeA[shapeA.size() - 1] == shapeB[shapeB.size() - 2]);

        Shape result(shapeA.size());
        for (size_t i = 0; i < result.size(); ++i)
        {
            if (i == result.size() - 1)
            {
                result[i] = shapeB[i];
            }
            else
            {
                result[i] = shapeA[i];
            }
        }

        // return std::nullopt;
        return vector<Shape>{result};
    }

} // namespace infini