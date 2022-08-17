from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter, CometWriter

from .experiment import Experiment

import subprocess


class SingleEnvExperiment(Experiment):
    """An Experiment object for training and testing agents that interact with one environment at a time."""

    def __init__(
        self,
        preset,
        env,
        name=None,
        train_steps=float("inf"),
        logdir="runs",
        quiet=False,
        render=False,
        write_loss=True,
        writer="tensorboard",
        options=None,
    ):
        self._name = name if name is not None else preset.name
        super().__init__(
            self._make_writer(logdir, self._name, env.name, write_loss, writer), quiet
        )
        self._logdir = logdir
        self._preset = preset
        self._agent = self._preset.agent(writer=self._writer, train_steps=train_steps)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1
        self._checkpoint_threshold = 0
        self._options = options

        if render:
            self._env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()

    def test(self, episodes=100, log=True):
        test_agent = self._preset.test_agent()
        returns = []
        for episode in range(episodes):
            episode_return = self._run_test_episode(test_agent)
            returns.append(episode_return)
            if log:
                self._log_test_episode(episode, episode_return)
        if log:
            self._log_test(returns)
        return returns

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        state = self._env.reset()
        in_option = False

        if self._options != None and self._options.initiate():
            in_option = True
            action = self._options.get_action()
        else:
            action = self._agent.act(state)

        print("in_option")
        print(in_option)
        print("action")
        print(action)
        returns = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            
            if in_option or self._options.initiate():
                in_option = True
                action = self._options.get_action()
            else:
                action = self._agent.act(state)

            state = self._env.step(action)
            if in_option: 
                if self._options.terminate():
                    in_option = False

            returns += state.reward
            self._frame += 1

            if self._frame >= self._checkpoint_threshold:  # checkpointing
                print("Saving Checkpoint")
                Experiment.save(self, "preset" + str(int(self._checkpoint_threshold)))
                #subprocess.call(["sh", "removeEvents.sh"])

                if self._frame >= 1e6:
                    self._checkpoint_threshold += 1e6  # continue by 1M's
                elif self._frame >= 1e5:
                    self._checkpoint_threshold += 1e5  # continue by 100k's
                else:
                    self._checkpoint_threshold += 1e4  # walk up  by 10k's

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        # log the results
        self._log_training_episode(returns, fps)

        # update experiment state
        self._episode += 1

    def _run_test_episode(self, test_agent):
        # initialize the episode
        state = self._env.reset()
        action, probs = test_agent.act(state)
        returns = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action, probs = test_agent.act(state)
            returns += state.reward

        return returns

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        if writer == "comet":
            return CometWriter(
                self, agent_name, env_name, loss=write_loss, logdir=logdir
            )
        return ExperimentWriter(
            self, agent_name, env_name, loss=write_loss, logdir=logdir
        )
