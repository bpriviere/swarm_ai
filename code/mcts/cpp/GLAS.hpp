#pragma once

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

  Eigen::VectorXf eval(const Eigen::VectorXf& input) const
  {
    Eigen::VectorXf result = input;
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

  Eigen::VectorXf eval(const std::vector<Eigen::Vector4f>& input) const
  {
    Eigen::VectorXf X = Eigen::VectorXf::Zero(m_rho.sizeIn(), 1);
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
    X.segment(m_ds_a.sizeOut(), m_ds_b.sizeOut()) = m_ds_b.eval(input_b);
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

class GLAS
{
public:
  GLAS(
    const YAML::Node& node,
    std::default_random_engine& gen)
    : m_phi_a()
    , m_rho_a()
    , m_ds_a(m_phi_a, m_rho_a)
    , m_phi_b()
    , m_rho_b()
    , m_ds_b(m_phi_b, m_rho_b)
    , m_psi()
    , m_glas(m_ds_a, m_ds_b, m_psi)
    , m_actions()
    , m_gen(gen)
  {
    m_phi_a = load(node, "model_team_a.phi");
    m_rho_a = load(node, "model_team_a.rho");
    m_phi_b = load(node, "model_team_b.phi");
    m_rho_b = load(node, "model_team_b.rho");
    m_psi = load(node, "psi");

    m_actions.resize(9);
    m_actions[0] << -1, -1;
    m_actions[1] << -1,  0;
    m_actions[2] << -1,  1;
    m_actions[3] <<  0, -1;
    m_actions[4] <<  0,  0;
    m_actions[5] <<  0,  1;
    m_actions[6] <<  1, -1;
    m_actions[7] <<  1,  0;
    m_actions[8] <<  1,  1;
  }

  const Eigen::Vector2f& computeAction(
    const std::vector<Eigen::Vector4f>& input_a,
    const std::vector<Eigen::Vector4f>& input_b,
    const Eigen::Vector4f& goal,
    bool deterministic) const
  {
    auto output = m_glas.eval(input_a, input_b, goal);
    int idx;
    if (deterministic) {
      output.maxCoeff(&idx);
    } else {
      // stochastic
      std::discrete_distribution<> dist(output.data(), output.data() + output.size());
      idx = dist(m_gen);
    }
    return m_actions[idx];
  }

private:
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

private:
  FeedForwardNN m_phi_a;
  FeedForwardNN m_rho_a;
  DeepSetNN m_ds_a;
  FeedForwardNN m_phi_b;
  FeedForwardNN m_rho_b;
  DeepSetNN m_ds_b;
  FeedForwardNN m_psi;
  DiscreteEmptyNet m_glas;

  std::vector<Eigen::Vector2f> m_actions;

  std::default_random_engine& m_gen;
};


template <std::size_t NumAttackers, std::size_t NumDefenders>
std::array<RobotAction, NumAttackers + NumDefenders> computeActionsWithGLAS(
  const GLAS& glas_a,
  const GLAS& glas_b,
  const GameState<NumAttackers, NumDefenders>& state,
  const Eigen::Vector2f& goal,
  const std::array<RobotType, NumAttackers>& attackerTypes,
  const std::array<RobotType, NumDefenders>& defenderTypes,
  std::default_random_engine& generator)
{
  std::array<RobotAction, NumAttackers + NumDefenders> action;
  std::vector<Eigen::Vector4f> input_a;
  std::vector<Eigen::Vector4f> input_b;
  Eigen::Vector4f relGoal;

  // evaluate glas for all team members of a
  for (size_t j = 0; j < NumAttackers; ++j) {
    // compute input_a
    input_a.clear();
    for (size_t j2 = 0; j2 < NumAttackers; ++j2) {
      if (j != j2) {
        Eigen::Vector4f relState;
        relState.segment(0,2) = state.attackers[j2].position - state.attackers[j].position;
        relState.segment(2,2) = state.attackers[j2].velocity - state.attackers[j].velocity;
        if (relState.segment(0,2).squaredNorm() <= attackerTypes[j].r_senseSquared) {
          input_a.push_back(relState);
        }
      }
    }
    // compute input_b
    input_b.clear();
    for (size_t j2 = 0; j2 < NumDefenders; ++j2) {
      Eigen::Vector4f relState;
      relState.segment(0,2) = state.defenders[j2].position - state.attackers[j].position;
      relState.segment(2,2) = state.defenders[j2].velocity - state.attackers[j].velocity;
      if (relState.segment(0,2).squaredNorm() <= attackerTypes[j].r_senseSquared) {
        input_b.push_back(relState);
      }
    }
    // compute relGoal
    relGoal.segment(0,2) = goal - state.attackers[j].position;
    relGoal.segment(2,2) = -state.attackers[j].velocity;

    // projecting goal to radius of sensing
    float alpha = sqrtf(relGoal.segment(0,2).squaredNorm() / attackerTypes[j].r_senseSquared);
    relGoal.segment(0,2) = relGoal.segment(0,2) / std::max(alpha, 1.0f);

    // evaluate GLAS
    auto a = glas_a.computeAction(input_a, input_b, relGoal, /*deterministic*/false);
    action[j] = a * attackerTypes[j].acceleration_limit;
  }

  // evaluate glas for all team members of b
  for (size_t j = 0; j < NumDefenders; ++j) {
    // compute input_a
    input_a.clear();
    for (size_t j2 = 0; j2 < NumAttackers; ++j2) {
      Eigen::Vector4f relState;
      relState.segment(0,2) = state.attackers[j2].position - state.defenders[j].position;
      relState.segment(2,2) = state.attackers[j2].velocity - state.defenders[j].velocity;
      if (relState.segment(0,2).squaredNorm() <= defenderTypes[j].r_senseSquared) {
        input_a.push_back(relState);
      }
    }
    // compute input_b
    input_b.clear();
    for (size_t j2 = 0; j2 < NumDefenders; ++j2) {
      if (j != j2) {
        Eigen::Vector4f relState;
        relState.segment(0,2) = state.defenders[j2].position - state.defenders[j].position;
        relState.segment(2,2) = state.defenders[j2].velocity - state.defenders[j].velocity;
        if (relState.segment(0,2).squaredNorm() <= defenderTypes[j].r_senseSquared) {
          input_b.push_back(relState);
        }
      }
    }
    // compute relGoal
    relGoal.segment(0,2) = goal - state.defenders[j].position;
    relGoal.segment(2,2) = -state.defenders[j].velocity;

    // projecting goal to radius of sensing
    float alpha = sqrtf(relGoal.segment(0,2).squaredNorm() / defenderTypes[j].r_senseSquared);
    relGoal.segment(0,2) = relGoal.segment(0,2) / std::max(alpha, 1.0f);

    // evaluate GLAS
    auto a = glas_b.computeAction(input_a, input_b, relGoal, /*deterministic*/false);
    action[NumAttackers + j] = a * defenderTypes[j].acceleration_limit;
  }

  return action;
}