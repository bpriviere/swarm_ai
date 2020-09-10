#pragma once

class FeedForwardNN
{
public:
  void addLayer(const Eigen::MatrixXf& weight, const Eigen::MatrixXf& bias)
  {
    m_layers.push_back({weight, bias});
  }

  Eigen::VectorXf eval(const Eigen::VectorXf& input) const
  {
    assert(m_layers.size() > 0);
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
    assert(m_layers.size() > 0);
    return m_layers[0].weight.cols();
  }

  size_t sizeOut() const
  {
    assert(m_layers.size() > 0);
    return m_layers.back().bias.size();
  }

  bool valid() const
  {
    return m_layers.size() > 0;
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

template<int InputDim>
class DeepSetNN
{
public:
  Eigen::VectorXf eval(const std::vector<Eigen::Matrix<float,InputDim,1>>& input) const
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

  FeedForwardNN& phi()
  {
    return m_phi;
  }

  FeedForwardNN& rho()
  {
    return m_rho;
  }

private:
  FeedForwardNN m_phi;
  FeedForwardNN m_rho;
};

template<int StateDim>
class DiscreteEmptyNet
{
public:
  typedef Eigen::Matrix<float, StateDim, 1> StateVector;

  Eigen::VectorXf eval(
    const std::vector<StateVector>& input_a,
    const std::vector<StateVector>& input_b,
    const StateVector& goal) const
  {
    Eigen::VectorXf X(m_psi.sizeIn());
    X.segment(0, m_ds_a.sizeOut()) = m_ds_a.eval(input_a);
    X.segment(m_ds_a.sizeOut(), m_ds_b.sizeOut()) = m_ds_b.eval(input_b);
    X.segment(m_ds_a.sizeOut()+m_ds_b.sizeOut(), 4) = goal;

    auto res = m_psi.eval(X);
    auto block = res.rightCols(res.cols()-1);
    // auto& block = res(Eigen::all,{1,2,3,4,5,6,7,8,9});
    // auto& block = res(seq(1,9));
    block = softmax(block);
    return res;
  }

  DeepSetNN<StateDim>& deepSetA()
  {
    return m_ds_a;
  }

  DeepSetNN<StateDim>& deepSetB()
  {
    return m_ds_b;
  }

  FeedForwardNN& psi()
  {
    return m_psi;
  }

private:
  Eigen::MatrixXf softmax(const Eigen::MatrixXf& m) const
  {
    auto exp = m.unaryExpr([](float x) {return std::exp(x);});
    return exp / exp.sum();
  }


private:
  DeepSetNN<StateDim> m_ds_a;
  DeepSetNN<StateDim> m_ds_b;
  FeedForwardNN m_psi;
};

template<int StateDim>
class GLAS
{
public:
  typedef Eigen::Matrix<float, StateDim, 1> StateVector;

  GLAS(std::default_random_engine& gen)
    : m_glas()
    , m_actions()
    , m_gen(gen)
  {
    m_actions.resize(9);
    m_actions[0] << -1/sqrtf(2), -1/sqrtf(2);
    m_actions[1] << -1,  0;
    m_actions[2] << -1/sqrtf(2),  1/sqrtf(2);
    m_actions[3] <<  0, -1;
    m_actions[4] <<  0,  0;
    m_actions[5] <<  0,  1;
    m_actions[6] <<  1/sqrtf(2), -1/sqrtf(2);
    m_actions[7] <<  1,  0;
    m_actions[8] <<  1/sqrtf(2),  1/sqrtf(2);
  }

  void load(const YAML::Node& node)
  {
    m_glas.deepSetA().phi().load(node, "model_team_a.phi");
    m_glas.deepSetA().rho().load(node, "model_team_a.rho");
    m_glas.deepSetB().phi().load(node, "model_team_b.phi");
    m_glas.deepSetB().rho().load(node, "model_team_b.rho");
    m_glas.psi().load(node, "psi");
  }

  const Eigen::Vector2f& computeAction(
    const std::vector<StateVector>& input_a,
    const std::vector<StateVector>& input_b,
    const StateVector& goal,
    bool deterministic) const
  {
    auto output = m_glas.eval(input_a, input_b, goal);
    output.rightCols(output.cols()-1);
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

  DiscreteEmptyNet<StateDim>& discreteEmptyNet()
  {
    return m_glas;
  }

  bool valid()
  {
    return m_glas.psi().valid();
  }

private:
  DiscreteEmptyNet<StateDim> m_glas;

  std::vector<Eigen::Vector2f> m_actions;

  std::default_random_engine& m_gen;
};


template<class Robot>
std::vector<typename Robot::Action> computeActionsWithGLAS(
  const GLAS<Robot::StateDim>& glas_a,
  const GLAS<Robot::StateDim>& glas_b,
  const GameState<Robot>& state,
  const Eigen::Matrix<float, Robot::StateDim, 1>& goal,
  const std::vector<typename Robot::Type>& attackerTypes,
  const std::vector<typename Robot::Type>& defenderTypes,
  std::default_random_engine& generator,
  bool deterministic)
{
  typedef Eigen::Matrix<float, Robot::StateDim, 1> StateVector;

  size_t NumAttackers = state.attackers.size();
  size_t NumDefenders = state.defenders.size();

  std::vector<typename Robot::Action> action(NumAttackers + NumDefenders);
  std::vector<StateVector> input_a;
  std::vector<StateVector> input_b;
  StateVector relGoal;

  // evaluate glas for all team members of a
  for (size_t j = 0; j < NumAttackers; ++j) {
    // compute input_a
    input_a.clear();
    for (size_t j2 = 0; j2 < NumAttackers; ++j2) {
      if (j != j2) {
        auto relState = state.attackers[j2].state - state.attackers[j].state;
        if (relState.template head<2>().squaredNorm() <= attackerTypes[j].r_senseSquared) {
          input_a.push_back(relState);
        }
      }
    }
    // compute input_b
    input_b.clear();
    for (size_t j2 = 0; j2 < NumDefenders; ++j2) {
      auto relState = state.defenders[j2].state - state.attackers[j].state;
      if (relState.template head<2>().squaredNorm() <= attackerTypes[j].r_senseSquared) {
        input_b.push_back(relState);
      }
    }
    // compute relGoal
    relGoal = goal - state.attackers[j].state;

    // projecting goal to radius of sensing
    float alpha = sqrtf(relGoal.template head<2>().squaredNorm() / attackerTypes[j].r_senseSquared);
    relGoal.template head<2>() = relGoal.template head<2>() / std::max(alpha, 1.0f);

    // evaluate GLAS
    auto a = glas_a.computeAction(input_a, input_b, relGoal, deterministic);
    action[j] = a * attackerTypes[j].actionLimit();
  }

  // evaluate glas for all team members of b
  for (size_t j = 0; j < NumDefenders; ++j) {
    // compute input_a
    input_a.clear();
    for (size_t j2 = 0; j2 < NumAttackers; ++j2) {
      auto relState = state.attackers[j2].state - state.defenders[j].state;
      if (relState.segment(0,2).squaredNorm() <= defenderTypes[j].r_senseSquared) {
        input_a.push_back(relState);
      }
    }
    // compute input_b
    input_b.clear();
    for (size_t j2 = 0; j2 < NumDefenders; ++j2) {
      if (j != j2) {
        auto relState = state.defenders[j2].state - state.defenders[j].state;
        if (relState.segment(0,2).squaredNorm() <= defenderTypes[j].r_senseSquared) {
          input_b.push_back(relState);
        }
      }
    }
    // compute relGoal
    relGoal = goal - state.defenders[j].state;

    // projecting goal to radius of sensing
    float alpha = sqrtf(relGoal.segment(0,2).squaredNorm() / defenderTypes[j].r_senseSquared);
    relGoal.segment(0,2) = relGoal.segment(0,2) / std::max(alpha, 1.0f);

    // evaluate GLAS
    auto a = glas_b.computeAction(input_a, input_b, relGoal, deterministic);
    action[NumAttackers + j] = a * defenderTypes[j].actionLimit();
  }

  return action;
}