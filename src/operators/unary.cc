#include "operators/unary.h"

namespace infini
{
    UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
        : OperatorObj(type, {input}, {output})
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        return {{A->getDims()}};
    }

    std::string UnaryObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                     std::optional<float> min, std::optional<float> max)
        : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
          maxValue(max)
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 clip 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13
        // =================================== 作业 ===================================
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        auto output_dim = input_dim;
        int rank = A->getRank();

        auto min = ClipObj::getMin();
        auto max = ClipObj::getMax();
        for (int i = 0; i < rank; ++i) {
            if (min.has_value() && max.has_value()) {
                output_dim[i] = std::fmin(max.value(), std::fmax(input_dim[i], min.value()));
            } else if (min.has_value()) {
                output_dim[i] = std::fmax(input_dim[i], min.value());
            } else if (max.has_value()) {
                output_dim[i] = std::fmin(input_dim[i], max.value());
            } else {
                output_dim[i] = input_dim[i];
            }
        }

        return vector<Shape>{output_dim};
    }

    std::string ClipObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        : OperatorObj(OpType::Cast, {input}, {output}), castType(type)
    {
        IT_ASSERT(checkValid(graph));
    }

    vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后, 输出 tensor 的数目和数据类型
        // REF_FILE: src/core/operator.cc
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        const auto A = inputs[0];
        // auto input_type = A->getDType();
        switch (this->getType())
        {
        case CastType::Float2Float16:
          return vector<DataType>{DataType::Float16};
        case CastType::Float2Int64:
          return vector<DataType>{DataType::Int64};
        case CastType::Float2Int32:
          return vector<DataType>{DataType::Int32};
        case CastType::Float2Int16:
          return vector<DataType>{DataType::Int16};
        case CastType::Float2Int8:
          return vector<DataType>{DataType::Int8};
        case CastType::Float2BFloat16:
          return vector<DataType>{DataType::BFloat16};

        case CastType::Int322Float:
          return vector<DataType>{DataType::Float32};
        case CastType::Int322Int8:
          return vector<DataType>{DataType::Int8};
        case CastType::Int322Int16:
          return vector<DataType>{DataType::Int16};
        case CastType::Int322Int64:
          return vector<DataType>{DataType::Int64};

        case CastType::Int162Float:
          return vector<DataType>{DataType::Float32};
        case CastType::Int162Int32:
          return vector<DataType>{DataType::Int32};

        case CastType::Int82Float:
          return vector<DataType>{DataType::Float32};
        case CastType::Int82Int16:
          return vector<DataType>{DataType::Int16};
        case CastType::Int82Int32:
          return vector<DataType>{DataType::Int32};

        case CastType::Uint82Float:
          return vector<DataType>{DataType::Float32};
        case CastType::Uint82Int32:
          return vector<DataType>{DataType::Int32};
        case CastType::Uint82Int64:
          return vector<DataType>{DataType::Int64};

        case CastType::Int642Int32:
          return vector<DataType>{DataType::Int32};
        case CastType::Int642Uint32:
          return vector<DataType>{DataType::UInt32};
        case CastType::Int642Float:
          return vector<DataType>{DataType::Float32};

        case CastType::Uint322Int64:
          return vector<DataType>{DataType::Int64};

        case CastType::Float162Float:
          return vector<DataType>{DataType::Float32};

        case CastType::BFloat162Float:
          return vector<DataType>{DataType::Float32};

        case CastType::Float2Float:
          return vector<DataType>{DataType::Float32};

        default:
          IT_TODO_HALT();
        }
    }

    optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        auto output_dim = input_dim;
        // int rank = A->getRank();

        // return std::nullopt;
        return vector<Shape>{output_dim};
    }

    std::string CastObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    DataType CastObj::getOutputDataType() const
    {
        switch (castType)
        {
        case CastType::Float2Float16:
            return DataType::Float16;
        case CastType::Float2Int64:
            return DataType::Int64;
        case CastType::Float2Int32:
            return DataType::Int32;
        case CastType::Float2Int16:
            return DataType::Int16;
        case CastType::Float2Int8:
            return DataType::Int8;
        case CastType::Int322Float:
            return DataType::Float32;
        case CastType::Int322Int8:
            return DataType::Int8;
        case CastType::Int322Int16:
            return DataType::Int16;
        case CastType::Int162Float:
            return DataType::Float32;
        case CastType::Int162Int32:
            return DataType::Int32;
        case CastType::Int82Float:
            return DataType::Float32;
        case CastType::Int82Int16:
            return DataType::Int16;
        case CastType::Int82Int32:
            return DataType::Int32;
        case CastType::Uint82Float:
            return DataType::Float32;
        case CastType::Uint82Int32:
            return DataType::Int32;
        case CastType::Uint82Int64:
            return DataType::Int64;
        case CastType::Int322Int64:
            return DataType::Int64;
        case CastType::Int642Int32:
            return DataType::Int32;
        case CastType::Int642Uint32:
            return DataType::UInt32;
        case CastType::Int642Float:
            return DataType::Float32;
        case CastType::Uint322Int64:
            return DataType::Int64;
        case CastType::Float162Float:
            return DataType::Float32;
        case CastType::BFloat162Float:
            return DataType::Float32;
        case CastType::Float2BFloat16:
            return DataType::BFloat16;
        case CastType::Float2Float:
            return DataType::Float32;
        default:
            IT_TODO_HALT();
        }
    }
}; // namespace infini
