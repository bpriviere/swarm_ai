#include <iostream>
#include <array>


#include "monte_carlo_tree_search.hpp"

// using libMultiRobotPlanning::AStar;
// using libMultiRobotPlanning::Neighbor;
// using libMultiRobotPlanning::PlanResult;

struct State {
  enum class Cell {
    Empty,
    X,
    O,
  };
  enum class Turn {
    PlayerX,
    PlayerO,
  };

  std::array<std::array<Cell,3>,3> cells;
  Turn turn;
};

std::ostream& operator<<(std::ostream& os, const State& s) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      switch(s.cells[i][j]) {
        case State::Cell::Empty:
          os << " . ";
          break;
        case State::Cell::X:
          os << " X ";
          break;
        case State::Cell::O:
          os << " O ";
          break;
      }
    }
    os << "\n";
  }
  os << "turn: ";
  switch(s.turn) {
    case State::Turn::PlayerX:
      os << "X";
      break;
    case State::Turn::PlayerO:
      os << "O";
      break;
  }
  return os;
}

struct Action {
  int i;
  int j;
};

typedef std::pair<float, float> Reward;

Reward operator+=(Reward& r1, const Reward& r2) {
  r1.first += r2.first;
  r1.second += r2.second;
  return r1;
}

class Environment {
 public:
  Environment(std::default_random_engine& generator)
    : m_generator(generator)
  {
  }

  bool step(const State& state, const Action& action, State& nextState)
  {
    // check if this action is valid
    if (state.cells[action.i][action.j] != State::Cell::Empty) {
      return false;
    }
    if (isTerminal(state)) {
      return false;
    }

    // copy state and make changes
    nextState = state;
    auto& cell = nextState.cells[action.i][action.j];

    switch(state.turn) {
      case State::Turn::PlayerX:
        cell = State::Cell::X;
        nextState.turn = State::Turn::PlayerO;
        break;
      case State::Turn::PlayerO:
        cell = State::Cell::O;
        nextState.turn = State::Turn::PlayerX;
        break;
    }

    return true;
  }

  bool isTerminal(const State& state)
  {
    return computeReward(state) >= 0.0;
  }

  float rewardToFloat(const State& state, const Reward& reward)
  {
    if (state.turn == State::Turn::PlayerX) {
      return reward.first;
    }
    return reward.second;
  }

  void getPossibleActions(const State& state, std::vector<Action>& actions)
  {
    // We could filter here the "valid" actions, but this is also checked in the "step" function
    actions.resize(9);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        actions[i+3*j].i = i;
        actions[i+3*j].j = j;
      }
    }
  }

  Reward rollout(const State& state)
  {
    float reward = computeReward(state);
    if (reward < 0) {
      std::vector<Action> validActions;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (state.cells[i][j] == State::Cell::Empty) {
            validActions.push_back({i,j});
          }
        }
      }
      std::shuffle(validActions.begin(), validActions.end(), m_generator);

      State s = state;
      while (validActions.size() > 0) {
        const Action& action = validActions.back();
        step(s, action, s);
        validActions.pop_back();
        reward = computeReward(s);
        if (reward >= 0) {
          break;
        }
      }
    }
    return Reward(reward, 1-reward);
  }

// private:
  float computeReward(const State& state)
  {
    // X wins if it has a row, column, or diagonal
    if (state.cells[0][0] == State::Cell::X && state.cells[0][1] == State::Cell::X && state.cells[0][2] == State::Cell::X) return 1.0;
    if (state.cells[1][0] == State::Cell::X && state.cells[1][1] == State::Cell::X && state.cells[1][2] == State::Cell::X) return 1.0;
    if (state.cells[2][0] == State::Cell::X && state.cells[2][1] == State::Cell::X && state.cells[2][2] == State::Cell::X) return 1.0;
    
    if (state.cells[0][0] == State::Cell::X && state.cells[1][0] == State::Cell::X && state.cells[2][0] == State::Cell::X) return 1.0;
    if (state.cells[0][1] == State::Cell::X && state.cells[1][1] == State::Cell::X && state.cells[2][1] == State::Cell::X) return 1.0;
    if (state.cells[0][2] == State::Cell::X && state.cells[1][2] == State::Cell::X && state.cells[2][2] == State::Cell::X) return 1.0;

    if (state.cells[0][0] == State::Cell::X && state.cells[1][1] == State::Cell::X && state.cells[2][2] == State::Cell::X) return 1.0;
    if (state.cells[0][2] == State::Cell::X && state.cells[1][1] == State::Cell::X && state.cells[2][0] == State::Cell::X) return 1.0;

    // O wins if it has a row, column, or diagonal
    if (state.cells[0][0] == State::Cell::O && state.cells[0][1] == State::Cell::O && state.cells[0][2] == State::Cell::O) return 0.0;
    if (state.cells[1][0] == State::Cell::O && state.cells[1][1] == State::Cell::O && state.cells[1][2] == State::Cell::O) return 0.0;
    if (state.cells[2][0] == State::Cell::O && state.cells[2][1] == State::Cell::O && state.cells[2][2] == State::Cell::O) return 0.0;
    
    if (state.cells[0][0] == State::Cell::O && state.cells[1][0] == State::Cell::O && state.cells[2][0] == State::Cell::O) return 0.0;
    if (state.cells[0][1] == State::Cell::O && state.cells[1][1] == State::Cell::O && state.cells[2][1] == State::Cell::O) return 0.0;
    if (state.cells[0][2] == State::Cell::O && state.cells[1][2] == State::Cell::O && state.cells[2][2] == State::Cell::O) return 0.0;

    if (state.cells[0][0] == State::Cell::O && state.cells[1][1] == State::Cell::O && state.cells[2][2] == State::Cell::O) return 0.0;
    if (state.cells[0][2] == State::Cell::O && state.cells[1][1] == State::Cell::O && state.cells[2][0] == State::Cell::O) return 0.0;

    // not yet over if not all cells are occupied
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        if (state.cells[i][j] == State::Cell::Empty) {
          return -1.0;
        }
      }
    }
    // Draw if all states are occupied
    return 0.5;
  }

private:
  std::default_random_engine& m_generator;


};


int main(int argc, char* argv[]) {

  size_t num_nodes = 10000;

  // std::random_device r;
  // std::default_random_engine generator(r());
  std::default_random_engine generator(0);

  State state;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      state.cells[i][j] = State::Cell::Empty;
    }
  }
  state.turn = State::Turn::PlayerX;

  std::cout << state << std::endl;

  Environment env(generator);

  libMultiRobotPlanning::MonteCarloTreeSearch<State, Action, Reward, Environment> mcts(env, generator, num_nodes, 1.4);

  while (true) {
    Action action;
    bool success = mcts.search(state, action);
    if (!success) {
      break;
    }
    env.step(state, action, state);
    float f = env.computeReward(state);
    std::cout << state << "reward: " << f << std::endl;
    // Reward r;
    // assert(r.first == 0);
    // assert(r.second == 0);
    // int i;
    // for (i = 0; i < 100000; ++i) {
    //   r += env.rollout(state);
    // }
    // std::cout << i << " " << r.first / i << " " << r.second / i << std::endl;
  }

  return 0;
}
