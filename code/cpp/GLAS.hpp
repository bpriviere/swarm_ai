#pragma once

#include <unsupported/Eigen/MatrixFunctions>

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

  size_t sizeIn() const
  {
    return m_phi.sizeIn();
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

// template<int StateDim>
// class DiscreteEmptyNet
// {
// public:
//   typedef Eigen::Matrix<float, StateDim, 1> StateVector;

//   Eigen::VectorXf eval(
//     const std::vector<StateVector>& input_a,
//     const std::vector<StateVector>& input_b,
//     const StateVector& goal) const
//   {
//     Eigen::VectorXf X(m_psi.sizeIn());
//     X.segment(0, m_ds_a.sizeOut()) = m_ds_a.eval(input_a);
//     X.segment(m_ds_a.sizeOut(), m_ds_b.sizeOut()) = m_ds_b.eval(input_b);
//     X.segment(m_ds_a.sizeOut()+m_ds_b.sizeOut(), 4) = goal;

//     auto res = m_psi.eval(X);
//     auto block = res.tail(res.size()-1);
//     block = softmax(block);
//     res(0) = (tanh(res(0))+1)/2;

//     return res;
//   }

//   DeepSetNN<StateDim>& deepSetA()
//   {
//     return m_ds_a;
//   }

//   DeepSetNN<StateDim>& deepSetB()
//   {
//     return m_ds_b;
//   }

//   FeedForwardNN& psi()
//   {
//     return m_psi;
//   }

// private:
//   Eigen::MatrixXf softmax(const Eigen::MatrixXf& m) const
//   {
//     auto exp = m.unaryExpr([](float x) {return std::exp(x);});
//     return exp / exp.sum();
//   }


// private:
//   DeepSetNN<StateDim> m_ds_a;
//   DeepSetNN<StateDim> m_ds_b;
//   FeedForwardNN m_psi;
// };

template<class Robot>
class GLAS
{
public:
  typedef Eigen::Matrix<float, Robot::StateDim, 1> StateVector;

  GLAS(std::default_random_engine& gen)
    : m_gen(gen)
  {
  }

  Eigen::VectorXf eval(
    const std::vector<StateVector>& input_a,
    const std::vector<StateVector>& input_b,
    const StateVector& goal,
    float action_limit,
    bool deterministic) const
  {
    // evaluate deep sets
    Eigen::VectorXf policy_input(m_policy.sizeIn());
    policy_input.segment(0, m_ds_a.sizeOut()) = m_ds_a.eval(input_a);
    policy_input.segment(m_ds_a.sizeOut(), m_ds_b.sizeOut()) = m_ds_b.eval(input_b);
    policy_input.segment(m_ds_a.sizeOut()+m_ds_b.sizeOut(), m_ds_b.sizeIn()) = goal;

    // Eigen::VectorXf action(Robot::ActionDim);

    // Eigen::VectorXf action(2);
    Eigen::VectorXf action(3);

    // if (isGaussian()) {
      auto policy = m_policy.eval(policy_input);

      // auto mu = policy.segment<typename Robot::ActionDim>(0);

      // auto mu = policy.segment<2>(0);
      auto mu = policy.segment<3>(0);

      if (deterministic) {
        action = mu;
      } else {

        // auto logvar = policy.segment<Robot::ActionDim>(Robot::ActionDim);

        // auto logvar = policy.segment<2>(2);
        auto logvar = policy.segment<3>(3);

        auto sd = logvar.array().exp().sqrt();

        // for (int i = 0; i < Robot::ActionDim; ++i) {
        
        // for (int i = 0; i < 2; ++i) {
        for (int i = 0; i < 3; ++i) {
          
          std::normal_distribution<float> dist(mu(i),sd(i));
          action(i) = dist(m_gen);
        }
      }
    // } else {
    //   // evaluate psi to compute condition y
    //   auto y = m_psi.eval(psi_input);

    //   // evaluate decoder to compute distribution
    //   Eigen::VectorXf dec_input(m_decoder.sizeIn());
    //   int z_dim = m_decoder.sizeIn() - m_psi.sizeOut();
    //   auto z = dec_input.segment(0, z_dim);
    //   if (deterministic) {
    //     z.setZero();
    //   } else {
    //     std::normal_distribution<float> dist(0.0,1.0);
    //     for (int i = 0; i < z_dim; ++i) {
    //       z(i) = dist(m_gen);
    //     }
    //   }
    //   dec_input.segment(z_dim, y.size()) = y;
    //   action = m_decoder.eval(dec_input);
    // }

    // scale action
    // float action_norm = action.norm();
    // if (action_norm > action_limit) {
    //   action = action / action_norm * action_limit;
    // }

    // Robot.RobotTypeDoubleIntegrator2D robotType;
    // robotType.scaleAction(action)

    // // evaluate value
    // auto val = m_value.eval(y);
    // float value = (tanh(val(0))+1)/2;

    return action;
  }


  Eigen::VectorXf eval(
    const GameState<Robot>& state,
    const StateVector& goal,
    const typename Robot::Type& robotType,
    bool teamAttacker,
    size_t idx,
    bool deterministic) const
  {

    const size_t NumAttackers = state.attackers.size();
    const size_t NumDefenders = state.defenders.size();

    std::vector<StateVector> input_a;
    std::vector<StateVector> input_b;
    StateVector relGoal;

    const auto& my_state = teamAttacker ? state.attackers[idx].state : state.defenders[idx].state;

    // compute input_a
    for (size_t i = 0; i < NumAttackers; ++i) {
      if (    (!teamAttacker || i != idx) 
           && state.attackers[i].status == Robot::State::Status::Active) {
        auto relState = state.attackers[i].state - my_state;
        typename Robot::State robotRelState(relState);
        if (robotRelState.position().squaredNorm() <= robotType.r_senseSquared) {
          input_a.push_back(relState);
        }
      }
    }
    // compute input_b
    for (size_t i = 0; i < NumDefenders; ++i) {
      if (    (teamAttacker || i != idx)
           && state.defenders[i].status == Robot::State::Status::Active) {
        auto relState = state.defenders[i].state - my_state;
        typename Robot::State robotRelState(relState);
        if (robotRelState.position().squaredNorm() <= robotType.r_senseSquared) {
          input_b.push_back(relState);
        }
      }
    }
    // compute relGoal
    relGoal = goal - my_state;

    // projecting goal to radius of sensing
    typename Robot::State robotRelGoal(relGoal);
    float alpha = sqrtf(robotRelGoal.position().squaredNorm() / robotType.r_senseSquared);
    robotRelGoal.position() = robotRelGoal.position() / std::max(alpha, 1.0f);

    // evaluate GLAS
    auto result = eval(input_a, input_b, robotRelGoal.state, robotType.actionLimit(), deterministic);
    robotType.scaleAction(result);
    return result;
  }

  auto& deepSetA()
  {
    return m_ds_a;
  }

  auto& deepSetB()
  {
    return m_ds_b;
  }

  // auto& psi()
  // {
  //   return m_psi;
  // }

  // auto& encoder()
  // {
  //   return m_encoder;
  // }

  // auto& decoder()
  // {
  //   return m_decoder;
  // }

  // auto& value()
  // {
  //   return m_value;
  // }

  auto& policy()
  {
    return m_policy;
  }

  bool valid() const
  {
    return m_policy.valid();
  }

// private:
//   bool isGaussian() const
//   {
//     return !m_decoder.valid();
//   }

private:
  std::default_random_engine& m_gen;

  DeepSetNN<Robot::StateDim> m_ds_a;
  DeepSetNN<Robot::StateDim> m_ds_b;
  // FeedForwardNN m_psi;
  // FeedForwardNN m_encoder;
  // FeedForwardNN m_decoder;
  // FeedForwardNN m_value;
  FeedForwardNN m_policy;

};

// template<class Robot>
// class GLAS
// {
// public:
//   typedef Eigen::Matrix<float, Robot::StateDim, 1> StateVector;

//   GLAS(std::default_random_engine& gen)
//     : m_glas()
//     , m_actions()
//     , m_gen(gen)
//   {
//     m_actions.resize(9);
//     m_actions[0] << -1/sqrtf(2), -1/sqrtf(2);
//     m_actions[1] << -1,  0;
//     m_actions[2] << -1/sqrtf(2),  1/sqrtf(2);
//     m_actions[3] <<  0, -1;
//     m_actions[4] <<  0,  0;
//     m_actions[5] <<  0,  1;
//     m_actions[6] <<  1/sqrtf(2), -1/sqrtf(2);
//     m_actions[7] <<  1,  0;
//     m_actions[8] <<  1/sqrtf(2),  1/sqrtf(2);
//   }

//   void load(const YAML::Node& node)
//   {
//     m_glas.deepSetA().phi().load(node, "model_team_a.phi");
//     m_glas.deepSetA().rho().load(node, "model_team_a.rho");
//     m_glas.deepSetB().phi().load(node, "model_team_b.phi");
//     m_glas.deepSetB().rho().load(node, "model_team_b.rho");
//     m_glas.psi().load(node, "psi");
//   }

//   Eigen::Vector2f computeAction(
//     const typename Robot::State& state,
//     const typename Robot::Type& robotType,
//     float dt,
//     const std::vector<StateVector>& input_a,
//     const std::vector<StateVector>& input_b,
//     const StateVector& goal,
//     bool deterministic) const
//   {

//     auto nn = m_glas.eval(input_a, input_b, goal);
//     auto output = nn.tail(nn.size()-1);

//     // mask actions that are invalid by setting their weight to 0
//     bool anyValid = false;
//     typename Robot::State nextState;
//     for (size_t i = 0; i < m_actions.size(); ++i) {
//       auto a = m_actions[i] * robotType.actionLimit();
//       robotType.step(state, a, dt, nextState);
//       if (!robotType.isStateValid(nextState)) {
//         output(i) = 0;
//       } else {
//         anyValid = true;
//       }
//     }

//     if (!anyValid) {
//       return robotType.invalidAction;
//     }

//     int idx;
//     if (deterministic) {
//       output.maxCoeff(&idx);
//     } else {
//       // stochastic
//       std::discrete_distribution<> dist(output.data(), output.data() + output.size());
//       idx = dist(m_gen);
//     }
//     return m_actions[idx] * robotType.actionLimit();
//   }

//   DiscreteEmptyNet<Robot::StateDim>& discreteEmptyNet()
//   {
//     return m_glas;
//   }

//   bool valid()
//   {
//     return m_glas.psi().valid();
//   }

// private:
//   DiscreteEmptyNet<Robot::StateDim> m_glas;

//   std::vector<Eigen::Vector2f> m_actions;

//   std::default_random_engine& m_gen;
// };


// template<class Robot>
// std::vector<typename Robot::Action> computeActionsWithGLAS(
//   const GLAS<Robot>& glas_a,
//   const GLAS<Robot>& glas_b,
//   const GameState<Robot>& state,
//   const Eigen::Matrix<float, Robot::StateDim, 1>& goal,
//   const std::vector<typename Robot::Type>& attackerTypes,
//   const std::vector<typename Robot::Type>& defenderTypes,
//   bool deterministic)
// {
//   size_t NumAttackers = state.attackers.size();
//   size_t NumDefenders = state.defenders.size();

//   std::vector<typename Robot::Action> action(NumAttackers + NumDefenders);

//   // evaluate glas for all team members of a
//   for (size_t j = 0; j < NumAttackers; ++j) {
//     auto result_a = glas_a.eval(state, goal, attackerTypes[j], true, j, deterministic);
//     action[j] = std::get<1>(result_a);
//   }

//   // evaluate glas for all team members of b
//   for (size_t j = 0; j < NumDefenders; ++j) {
//     auto result_b = glas_b.eval(state, goal, defenderTypes[j], false, j, deterministic);
//     action[NumAttackers + j] = std::get<1>(result_b);
//   }

//   return action;
// }