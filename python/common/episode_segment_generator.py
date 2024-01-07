# Copyright 2019 - 2024 The ROMAN ROADS Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np


def segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()

    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    save = False
    player_score = 0
    opponent_score = 0
    wins, losses, ties, games_total, total_player, total_opponent = 0, 0, 0, 0, 0, 0

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        if t > 0 and t % horizon == 0:
            # Note: this is synchronously called when every episode batch size is reached,
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}

            _, vpred = pi.act(stochastic, ob)
            ep_rets = []
            ep_lens = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)

        env.render()

        if rew > 0:
            player_score += abs(rew)
        else:
            opponent_score += abs(rew)

        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1

        # Note: this is asynchronously called every time when an episode is finished
        if new:
            print(format("[INFO] End of game: score %d - %d" % (player_score, opponent_score)))
            games_total += 1

            if player_score > opponent_score:
                wins += 1
            elif opponent_score > player_score:
                losses += 1
            else:
                ties += 1

            total_player += player_score
            total_opponent += opponent_score

            player_score, opponent_score = 0, 0

            print(format("[INFO ] Games played - %d, wins - %d, losses - %d, ties - %d"
                         % (games_total, wins, losses, ties)))

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0

            # Note this line should block until the game env got reset
            ob = env.reset()
        t += 1
