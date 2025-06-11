# Project title : The Vulture Patrol in Starcraft 2
# Author : Lee hae won, Yim yong hwan
# Final update : 2023.06.19
import random
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features, actions
from pysc2.agents import base_agent
from torch.utils.tensorboard import SummaryWriter

APM = 7          # Actions Per Minute : 모든 작업을 계산
VISUALIZE = False
REALTIME = False
MAX_EPISODE = 10000   # init = 300
REWARD = np.array([0.5, -0.2, 1, -1])

# Normalize parameter
_X_NORMALIZE = 56
_Y_NORMALIZE = 40
_CUR_HP_NORMALIZE = 75
_COOL_DOWN_NORMALIZE = 27

_PLAYER_FRIENDLY = 1  # Vulture; use for player_relative [0,4]
_PLAYER_HOSTILE = 4   # Zerglings
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id  # use for actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id # use for actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
_NOT_QUEUED = [0]
_QUEUED = [1]
MAPNAME = 'DefeatZerglingsAndBanelings' # 맵이름과 다르게 실제 게임은 Vulture vs Zergling
UNLIMIT = 0
SCREEN_SIZE = 64
MINIMAP_SIZE = 48
MIN_X, MAX_X = 3, 59
MIN_Y, MAX_Y = 3, 43
lossHist =[]
RESCALE_MIN_X, RESCALE_MAX_X = 3, 59
RESCALE_MIN_Y, RESCALE_MAX_Y = 3, 59
MAX_NORM = 5 # for gradient clipping
writer = SummaryWriter()

# Action Define.
UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'
UP_R, UP_L, DOWN_R, DOWN_L = 'up_r', 'up_l', 'down_r', 'down_l'
ATTACK = 'attack'
action_list = [UP, DOWN, LEFT, RIGHT, UP_L, UP_R, DOWN_R, DOWN_L, ATTACK]
MOVE_ACTIONS = [actions.FUNCTIONS.Move_screen.id, actions.FUNCTIONS.Move_minimap.id]
players = [sc2_env.Agent(sc2_env.Race.terran)]
interface = features.AgentInterfaceFormat(
    feature_dimensions=features.Dimensions(
        screen=SCREEN_SIZE, minimap=MINIMAP_SIZE), use_feature_units=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, dimState, dimAction, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(dimState, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dimAction)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out
        
class ReplayBuffer:
    def __init__(self, maxlen, dimState):
        self.maxlen = maxlen
        self.dimState = dimState
        self.state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.act_buff = np.zeros((self.maxlen, 1), dtype=np.int64)
        self.rew_buff = np.zeros(self.maxlen, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.mask_buff = np.zeros(self.maxlen, dtype=np.uint8)
        self.filled = 0
        self.position = 0
        
    def push(self, s, a, r, sp, mask):
        self.state_buff[self.position] = s
        self.act_buff[self.position] = a
        self.rew_buff[self.position] = r
        self.next_state_buff[self.position] = sp
        self.mask_buff[self.position] = mask
        self.position = (self.position + 1) % self.maxlen
        if self.filled < self.maxlen:
            self.filled += 1
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.filled), size=batch_size,
                               replace=True)
        s = torch.FloatTensor(self.state_buff[idx])
        a = torch.LongTensor(self.act_buff[idx])
        r = torch.FloatTensor(self.rew_buff[idx])
        sp = torch.FloatTensor(self.next_state_buff[idx])
        mask = torch.Tensor(self.mask_buff[idx])
        return s, a, r, sp, mask
        
    def __len__(self):
        return self.filled
        
    def clear(self):
        self.state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.act_buff = np.zeros((self.maxlen, 1), dtype=np.int64)
        self.rew_buff = np.zeros(self.maxlen, dtype=np.float32)
        self.next_state_buff = np.zeros((self.maxlen, self.dimState), dtype=np.float32)
        self.mask_buff = np.zeros(self.maxlen, dtype=np.uint8)
        self.filled_i = 0
        self.curr_i = 0

class Agent(base_agent.BaseAgent):
    def __init__(self, env):
        super(Agent, self).__init__()
        # Environment information
        self.env = env
        self.dimState = 6  # State 설정에 따라 변경
        self.dimAction = 9  # Action 설정에 따라 변경
        self.gamma = 0.99  # discount factor
        # Q network initialization
        self.hidden_dim = 128
        # QNet = MLP_QNet
        QNet = DQN
        self.qnet = QNet(self.dimState, self.dimAction, self.hidden_dim)
        self.target_qnet = QNet(self.dimState, self.dimAction, self.hidden_dim)
        self.hard_update()
        # Experience replay buffer
        self.memory = ReplayBuffer(500000, self.dimState)
        self.batch_size = 64
        # Learning parameters
        self.tau = 1e-3  # target network update rate
        self.lr = 1e-3  # learning rate
        self.optimizer = Adam(self.qnet.parameters(), lr=self.lr, weight_decay=0)
        # epsilon annealing parameters
        self.eps = 1.0
        self.eps_end = 0.1
        self.eps_step = 0.9997
        #self.eps_step = 1/MAX_EPISODE
        self.maxLoss = 0
        self.N_STEP = 0
        
    def save(self, fname):
        torch.save(self.qnet.state_dict(), fname)
        
    def load(self, fname):
        self.qnet.load_state_dict(torch.load(fname))
        
    # Epsilon-greedy policy
    def getAction(self, state):
        p = np.random.random()
        if p < self.eps:
            if np.random.random() < 0.1:
                return 8
            else:
                return np.random.randint(self.dimAction)
        else:
            return self.qnet(state).argmax().item()
            
    def hard_update(self):
        for target, source in zip(self.target_qnet.parameters(),
                                  self.qnet.parameters()):
            target.data.copy_(source.data)
                                      
    def soft_update(self):
        for target, source in zip(self.target_qnet.parameters(),
                                  self.qnet.parameters()):
            average = target.data * (1. - self.tau) + source.data * self.tau
            target.data.copy_(average)
                                      
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        s, a, r, sp, done = self.memory.sample(self.batch_size)
        out = self.qnet.forward(s)
        pred = out.gather(1, a)
        out_tar = self.target_qnet.forward(sp)
        target = r.unsqueeze(1) + self.gamma * out_tar.max(1)[0].unsqueeze(1) * (1 - done.unsqueeze(1))
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), MAX_NORM) # gradient clipping
        self.optimizer.step()
        self.optimizer.zero_grad()
        l = loss.item()
        lossHist.append(l)
        return l

    def act(self, action, position, obs):
        re_position = ((position[0] * _X_NORMALIZE / 2 + 31), (position[1] * _Y_NORMALIZE / 2 + 23))
        if action == UP:
            target = (re_position[0], re_position[1] - 10)
        elif action == DOWN:
            target = (re_position[0], re_position[1] + 10)
        elif action == LEFT:
            target = (re_position[0] - 10, re_position[1])
        elif action == RIGHT:
            target = (re_position[0] + 10, re_position[1])
        elif action == UP_R:
            target = (re_position[0] + 7, re_position[1] - 7)
        elif action == UP_L:
            target = (re_position[0] - 7, re_position[1] - 7)
        elif action == DOWN_R:
            target = (re_position[0] + 7, re_position[1] + 7)
        elif action == DOWN_L:
            target = (re_position[0] - 7, re_position[1] + 7)
        elif action == ATTACK:
            attack_target = (int(re_position[0]), int(re_position[1]))
            if not (MIN_X <= attack_target[0] <= MAX_X and MIN_Y <= attack_target[1] <= MAX_Y):
                attack_target = (max(attack_target[0], MIN_X), max(attack_target[1], MIN_Y))
                attack_target = (min(attack_target[0], MAX_X), min(attack_target[1], MAX_Y))
            if _ATTACK_SCREEN in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, attack_target])
            else:
                return actions.FUNCTIONS.no_op()
        else:
            raise ValueError('Invalid action')
        if not (MIN_X <= target[0] <= MAX_X and MIN_Y <= target[1] <= MAX_Y):
            target = (max(target[0],MIN_X),max(target[1],MIN_Y))
            target = (min(target[0],MAX_X),min(target[1],MAX_Y))
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FUNCTIONS.no_op()

    def preprocess_observation(self, obs):
        # [백그라운드, 자신, 동맹, 중립, 적] 유닛을 각각 나타내는 [0, 4]의 값을 취함.
        # "player_relative == 1" : 자신에 대한 좌표를 받아옴. 만약 4로 두면 저글링의 좌표를 받을 수 있음.
        # 가능한 action 확인을 위한 임시 코드
        # for action in obs.observation.available_actions:
        # print(actions.FUNCTIONS[action])
        player_relative = obs.observation.feature_screen.player_relative  # player_relative : 어떤 유닛이 우호적인지 적대적인지.
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        if not player_y.any():
            return None
        friendly_position = (int(player_x.mean()), int(player_y.mean()))
        norm_friendly_position = ((friendly_position[0]-31)/_X_NORMALIZE * 2, (friendly_position[1]-23)/_Y_NORMALIZE * 2)

        player_y, player_x = (player_relative == _PLAYER_HOSTILE).nonzero()
        player_x -= friendly_position[0]
        player_y -= friendly_position[1]
        closest_idx = np.argmin(player_x ** 2 + player_y ** 2)
        hostile_position = (player_x[closest_idx], player_y[closest_idx])
        norm_hostile_position = (hostile_position[0] / _X_NORMALIZE * 2, hostile_position[1] / _Y_NORMALIZE * 2)

        cur_hp = int(obs.observation.feature_units[obs.observation.feature_units[:, 0] == 1135, 2])
        norm_cur_hp = cur_hp / _CUR_HP_NORMALIZE * 2 - 1
        cooldown = int(obs.observation.feature_units[obs.observation.feature_units[:, 0] == 1135, 25])
        norm_cooldown = cooldown / _COOL_DOWN_NORMALIZE * 2 - 1
        preprocessed = norm_friendly_position + norm_hostile_position + (norm_cur_hp, norm_cooldown)

        return preprocessed

    def get_reward(self, past_obs, cur_obs):
        # obs에서 feature 뽑아와서 reward 계산
        # 공격량, 피해량, 킬, 데스
        feature = np.zeros(4)
        feature[0:2] = (cur_obs.score_by_vital - past_obs.score_by_vital)[0:2, 0]  # 딜량
        feature[2] = cur_obs.score_cumulative.total_value_units - past_obs.score_cumulative.total_value_units  # 킬 점수
        feature[3] = cur_obs.player.army_count - past_obs.player.army_count  # 남은 유닛 수
        feature[feature != 0] = 1
        if feature[1] == 1:  # 피격 시 현재 hp에 비례한 공격 reward 얻음(피 작으면 공격보다 도망이 좋게)
            cur_hp = int(past_obs.feature_units[past_obs.feature_units[:, 0] == 1135, 2])
            coef = cur_hp / _CUR_HP_NORMALIZE  # 현재 체력 / 최대 체력
            feature[0] *= coef
        reward = sum(feature * REWARD) - APM / 700
        return reward

    def runEpisode(self, test=False):
        self.setup(self.env.observation_spec(), self.env.action_spec())
        timestep = self.env.reset()  # 환경 초기화, timestep = environment
        s = self.preprocess_observation(timestep[0])
        self.reset()
        done = False
        rewards = []
        while True:
            s_tensor = torch.FloatTensor(s)
            if test:
                a = self.qnet(s_tensor).argmax().item()
            else:
                a = self.getAction(s_tensor)
            next_timestep = self.env.step([self.act(action_list[a], s[0:2], timestep[0])])  # add timestep[0] in act() fuction; add obs.
            sp = self.preprocess_observation(next_timestep[0])
            r = self.get_reward(timestep[0].observation,next_timestep[0].observation)     ### 아직 reward 설정 못함 -> reward 함수를 통해 설정 -> 전 step과 현재 step의 observation 가지고 결정
            done = next_timestep[0].last()
            super(Agent, self).step(timestep[0])    # timestep[0] = obs 에 내장되어 있는 reward 업데이트, 그러나 뭐라고 내장되어 있는지 모르겠음
            rewards.append(r)
            if not test:
                if done:
                    self.memory.push(s, a, r, s, done)
                else:
                    self.memory.push(s, a, r, sp, done)
                l = self.train()
                if self.N_STEP >= self.batch_size:
                    writer.add_scalar('Loss', l, self.N_STEP-self.batch_size)
                self.N_STEP += 1
                #self.soft_update()
                if self.N_STEP % 5000 == 0:
                    self.hard_update()
            timestep = next_timestep
            s = sp
            if done:
                if test:
                    score = timestep[0].observation.score_cumulative.score
                    return rewards, score
                break
        return rewards
        
    def runTest(self):
        rewards, score = self.runEpisode(test=True)
        ret = sum(rewards)
        nStep = len(rewards)
        print("Test episode, return = %.1f in %d steps" % (ret, nStep))
        return ret
        
    # Run multiple episodes to train the agent
    # and give a learning plot
    def runMany(self, nEpisode, fname):
        retHist = []
        testHist = []
        max5 = 0
        cnt = 0
        for ep_i in range(nEpisode):
            rewards = self.runEpisode()
            self.eps = max(self.eps_end, self.eps * self.eps_step)
            #self.eps = max(self.eps_end, self.eps - self.eps_step)
            ret = sum(rewards)
            writer.add_scalar('Returns', ret, ep_i)
            nStep = len(rewards)
            print("Train episode i=%d, return = %.1f in %d steps, eps = %.4f" % (ep_i, ret, nStep, self.eps))
            retHist.append(ret)
            if ep_i > 4:
                avg5 = sum(retHist[-5:]) / 5  # average return of the last 5 episodes
                if avg5 > max5 and max(retHist[-5:]) == retHist[-1]:
                    max5 = avg5
                    print("iter %d, avg5 = %.1f updated" % (ep_i, avg5))
                    if avg5 > 10:
                        self.save(str(cnt%3)+fname)
                        cnt += 1
            if ep_i % 100 == 99:
                test_ret = self.runTest()
                testHist.append(test_ret)

        print(max5)
        self.save(fname)
        self.plotReturn(retHist, 10, 'Online Reward')
        self.plotReturn(testHist, 0, 'Offline Reward')
        self.plotReturn(lossHist, 100, 'Loss')

    def testMany(self, nEpisode, fname):
        retHist = []
        stepHist = []
        scoreHist = []
        for ep_i in range(nEpisode):
            rewards, score = self.runEpisode(test=True)
            ret = sum(rewards)
            nStep = len(rewards)
            retHist.append(ret)
            stepHist.append(nStep)
            scoreHist.append((score))
            writer.add_scalar("Returns", ret, ep_i)
            writer.add_scalar("nSteps", nStep, ep_i)
            writer.add_scalar('Score', score, ep_i)
        retMean = np.mean(retHist)
        retStd = np.std(retHist)
        stepMean = np.mean(stepHist)
        stepStd = np.std(stepHist)
        scoreMean = np.mean(scoreHist)
        scoreStd = np.std(scoreHist)
        print("The performance of algorithm %s" % (fname))
        print("Mean of rewards: %.4f" % (retMean))
        print("Standard deviation of rewards: %.4f" % (retStd))
        print("Mean of number of steps: %.4f" % (stepMean))
        print("Standard deviation of number of steps: %.4f" % (stepStd))
        print("Mean of score: %.4f" % (scoreMean))
        print("Standard deviation of score: %.4f" % (scoreStd))

    def plotReturn(self, retHist, m=0, title='Title'):
        plt.plot(retHist)
        if m > 1 and m < len(retHist):
            cumsum = [0]
            movAvg = []
            for i, x in enumerate(retHist, 1):
                cumsum.append(cumsum[i - 1] + x)
                if i < m:
                    i0 = 0
                    n = i
                else:
                    i0 = i - m
                    n = m
                ma = (cumsum[i] - cumsum[i0]) / n
                movAvg.append(ma)
            plt.plot(movAvg)
        plt.title(title)
        plt.show()

def trainAgentDQN(fname):
    try:
        with sc2_env.SC2Env(map_name=MAPNAME, players=players,
                            agent_interface_format=interface,
                            step_mul=APM, game_steps_per_episode=UNLIMIT,
                            visualize=VISUALIZE, realtime=REALTIME) as env:
            agent = Agent(env)
            agent.runMany(MAX_EPISODE, fname=fname)
    except KeyboardInterrupt:
        pass

def trainContinuingAgentDQN(fname):
    try:
        with sc2_env.SC2Env(map_name=MAPNAME, players=players,
                            agent_interface_format=interface,
                            step_mul=APM, game_steps_per_episode=UNLIMIT,
                            visualize=VISUALIZE, realtime=REALTIME) as env:
            agent = Agent(env)
            agent.load(fname)
            agent.eps = 0.8
            agent.runMany(MAX_EPISODE, fname=fname)
    except KeyboardInterrupt:
        pass

def testBestAgentDQN(fname):
    try:
        with sc2_env.SC2Env(map_name=MAPNAME, players=players,
                            agent_interface_format=interface,
                            step_mul=APM, game_steps_per_episode=UNLIMIT,
                            visualize=VISUALIZE, realtime=REALTIME) as env:
            agent = Agent(env)
            agent.load(fname)
            agent.runTest()
    except KeyboardInterrupt:
        pass

def testManyAgentDQN(fname):
    try:
        with sc2_env.SC2Env(map_name=MAPNAME, players=players,
                            agent_interface_format=interface,
                            step_mul=APM, game_steps_per_episode=UNLIMIT,
                            visualize=VISUALIZE, realtime=REALTIME) as env:
            agent = Agent(env)
            agent.load(fname)
            agent.testMany(100, fname=fname)
    except KeyboardInterrupt:
        pass

def main(args):
    #trainAgentDQN('DQN.pt')
    #trainContinuingAgentDQN('DQN.pt')
    #testBestAgentDQN('DQN.pt')
    testManyAgentDQN('DQN25.pt')

app.run(main)
