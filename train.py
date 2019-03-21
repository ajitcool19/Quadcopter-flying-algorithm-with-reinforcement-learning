import sys
import pandas as pd
from agents.policy_search import PolicySearch_Agent
from agents.agent import DDGP
from task import Task
import tensorflow as tf
import numpy as np

num_episodes = 1000
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = DDGP(task) 
best_score=[0]


with tf.device('/device:GPU:0'):
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        score=0
        while True:
            action = agent.act(state) 
            next_state, reward, done = task.step(action)
            score=score+reward
            agent.step(action, reward, next_state, done)
            state = next_state
            if done:
                print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                    i_episode,score,max(best_score)), end="\n")
                best_score.append(score)
                break
        sys.stdout.flush()
