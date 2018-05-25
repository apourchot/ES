import gym


class Env():

    """
    Simple wrapper for OpenAI gym
    """

    def __init__(self, name):

        self.env = gym.make(name)

    def close(self):
        self.env.close()

    def get_action_space(self):
        """
        Returns the structure and size of the action space
        """

        space = self.env.action_space
        name = str(space).split('(')[0]
        if name == 'Box':
            size = space.shape
        else:
            size = space.n
        return (name, size)

    def get_obs_space(self):
        """
        Returns the structure and size of the action space
        """

        space = self.env.observation_space
        name = str(space).split('(')[0]
        if name == 'Box':
            size = space.shape
        else:
            size = space.n
        return (name, size)

    def eval(self, nn, render=False, t_max=200):
        """
        Computes the score of an agent on a run
        """

        score = 0
        obs = self.env.reset()

        for t in range(t_max):

            # get next action and act
            action = nn.get_action(obs)
            obs, reward, done, info = self.env.step(action)
            score += reward

            if(render):
                self.env.render()

            if done:
                self.env.reset()
                return score

        return score
