#include <iostream>
#include <array>
#include <Eigen/Dense>
#include <fstream>
#include <bitset>
#include <unordered_map>
#include <random>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

namespace YAML {
template<>
struct convert<Eigen::MatrixXf> {
  static Node encode(const Eigen::MatrixXf& rhs) {
    Node node;
    // node.push_back(rhs.x);
    // node.push_back(rhs.y);
    // node.push_back(rhs.z);
    return node;
  }

  static bool decode(const Node& node, Eigen::MatrixXf& rhs) {
    if(!node.IsSequence()) {
      return false;
    }

    size_t numRows = node.size();
    size_t numCols = 1;
    if (node[0].IsSequence()) {
      // 2D matrix
      size_t numCols = node[0].size();
      rhs.resize(numRows, numCols);
      for (size_t r = 0; r < numRows; ++r) {
        for (size_t c = 0; c < numCols; ++c) {
          rhs(r,c) = node[r][c].as<float>();
        }
      }
    } else {
      // 1D matrix
      rhs.resize(numRows, 1);
      for (size_t r = 0; r < numRows; ++r) {
        rhs(r) = node[r].as<float>();
      }
    }
    // std::cout << numRows << " " << numCols << std::endl;

    // rhs.resize(numRows,numCols);
    // for (size_t r = 0; r < numRows; ++r) {
    //   for (size_t )
    // }

    // rhs.x = node[0].as<double>();
    // rhs.y = node[1].as<double>();
    // rhs.z = node[2].as<double>();
    return true;
  }
};
}

class FeedForwardNN
{
public:
  FeedForwardNN()
  {
  }

  void addLayer(const Eigen::MatrixXf& weight, const Eigen::MatrixXf& bias)
  {
    m_layers.push_back({weight, bias});
  }

  Eigen::MatrixXf eval(const Eigen::MatrixXf& input) const
  {
    Eigen::MatrixXf result = input;
    for (size_t i = 0; i < m_layers.size()-1; ++i) {
      const auto& l = m_layers[i];
      result = relu(l.weight * result + l.bias);
    }
    const auto& l = m_layers.back();
    result = l.weight * result + l.bias;
    return result;
  }

  size_t sizeIn() const
  {
    return m_layers[0].weight.cols();
  }

  size_t sizeOut() const
  {
    return m_layers.back().bias.size();
  }

private:
  Eigen::MatrixXf relu(const Eigen::MatrixXf& m) const
  {
    return m.cwiseMax(0);
  }

private:
  struct Layer {
    Eigen::MatrixXf weight;
    Eigen::MatrixXf bias;
  };
  std::vector<Layer> m_layers;
};

class DeepSetNN
{
public:
  DeepSetNN(
    const FeedForwardNN& phi,
    const FeedForwardNN& rho)
    : m_phi(phi)
    , m_rho(rho)
  {
  }

  Eigen::MatrixXf eval(const std::vector<Eigen::Vector4f>& input) const
  {
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(m_rho.sizeIn(), 1);
    for (const auto& i : input) {
      X += m_phi.eval(i);
    }
    return m_rho.eval(X);
  }

  size_t sizeOut() const
  {
    return m_rho.sizeOut();
  }

private:
  const FeedForwardNN& m_phi;
  const FeedForwardNN& m_rho;
};

class DiscreteEmptyNet
{
public:
  DiscreteEmptyNet(
    const DeepSetNN& ds_a,
    const DeepSetNN& ds_b,
    const FeedForwardNN& psi)
    : m_ds_a(ds_a)
    , m_ds_b(ds_b)
    , m_psi(psi)
  {
  }

  Eigen::VectorXf eval(
    const std::vector<Eigen::Vector4f>& input_a,
    const std::vector<Eigen::Vector4f>& input_b,
    const Eigen::Vector4f& goal) const
  {
    Eigen::VectorXf X(m_psi.sizeIn());
    X.segment(0, m_ds_a.sizeOut()) = m_ds_a.eval(input_a);
    X.segment(m_ds_a.sizeOut(), m_ds_b.sizeOut()) = m_ds_b.eval(input_a);
    X.segment(m_ds_a.sizeOut()+m_ds_b.sizeOut(), 4) = goal;

    auto res = m_psi.eval(X);
    return softmax(res);
  }

private:
  Eigen::MatrixXf softmax(const Eigen::MatrixXf& m) const
  {
    auto exp = m.unaryExpr([](float x) {return std::exp(x);});
    return exp / exp.sum();
  }


private:
  const DeepSetNN& m_ds_a;
  const DeepSetNN& m_ds_b;
  const FeedForwardNN& m_psi;
};

FeedForwardNN load(const YAML::Node& node, const std::string& name)
{
  FeedForwardNN nn;
  for (size_t l = 0; ; ++l) {
    std::string key1 = name + ".layers." + std::to_string(l) + ".weight";
    std::string key2 = name + ".layers." + std::to_string(l) + ".bias";
    if (node[key1] && node[key2]) {
      nn.addLayer(
        node[key1].as<Eigen::MatrixXf>(),
        node[key2].as<Eigen::MatrixXf>());
    } else {
      break;
    }
  }
  return nn;
}


int main(int argc, char* argv[]) {

  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(),"input file (YAML)")
    ("output,o", po::value<std::string>(&outputFile)->required(),"output file (YAML)");

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error& e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  YAML::Node cfg_nn = YAML::LoadFile(inputFile);

  // for (const auto& node : cfg_nn) {
  //   auto key = node.first.as<std::string>();

  //   Eigen::MatrixXf m = node.second.as<Eigen::MatrixXf>();
  //   std::cout << key << m.size() << std::endl;
  // }

  FeedForwardNN phi_a = load(cfg_nn, "model_team_a.phi");
  FeedForwardNN rho_a = load(cfg_nn, "model_team_a.rho");
  DeepSetNN ds_a(phi_a, rho_a);

  FeedForwardNN phi_b = load(cfg_nn, "model_team_b.phi");
  FeedForwardNN rho_b = load(cfg_nn, "model_team_b.rho");
  DeepSetNN ds_b(phi_b, rho_b);

  FeedForwardNN psi = load(cfg_nn, "psi");

  DiscreteEmptyNet glas(ds_a, ds_b, psi);

  std::vector<Eigen::Vector4f> input_a(1);
  input_a[0] << 0,0,0,0;

  std::vector<Eigen::Vector4f> input_b;

  Eigen::Vector4f goal;
  goal << 0,0,0,0;

  auto output = glas.eval(input_a, input_b, goal);

  std::vector<Eigen::Vector2f> actions(9);
  actions[0] << -1, -1;
  actions[1] << -1,  0;
  actions[2] << -1,  1;
  actions[3] <<  0, -1;
  actions[4] <<  0,  0;
  actions[5] <<  0,  1;
  actions[6] <<  1, -1;
  actions[7] <<  1,  0;
  actions[8] <<  1,  1;

  // deterministic
  int idx;
  output.maxCoeff(&idx);

  // stochastic
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(output.data(), output.data() + output.size());
  idx = d(gen);

  std::cout << output << " " << idx << " " << actions[idx] << std::endl;

  return 0;
}
