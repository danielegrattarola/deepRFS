from joblib import Parallel, delayed
import numpy as np
import imageio, time


def evaluate_policy(mdp, policy, metric='cumulative', n_episodes=1,
                    video=False, save_video=False,
                    save_path='', append_filename='', n_jobs=1,
                    initial_actions=None, eval_epsilon=0.05, clip=False):
    """
        This function evaluates a policy on the given environment w.r.t.
        the specified metric by executing multiple episode, using the
        provided feature extraction model to encode states.
        Params:
            mdp (object): the environment on which to run.
            policy (object): a policy object (method draw_action is
                expected).
            nn_stack (object): the feature extraction model (method
                s_features is expected).
            metric (string, 'cumulative'): the evaluation metric
                ['discounted', 'average', 'cumulative']
            n_episodes (int, 1): the number of episodes to run.
            video (bool, False): whether to render the environment.
            save_video (bool, False): whether to save the video of the
                evaluation episodes.
            save_path (string, ''): where to save videos of evaluation episodes.
            n_jobs (int, 1): the number of processes to use for evaluation
                (leave default value if the feature extraction model runs
                on GPU).
            initial_actions (list, None): actions to use to force start the
                episode (useful in environments that require a specific action
                to start the episode, like some Atari environments)
        Return:
            metric (float): the average of the selected evaluation metric.
            metric_confidence (float): 95% confidence level for the
                provided metric.
            steps (float): the average number of steps in an episode.
            steps_confidence (float): 95% confidence level for the number
                of steps.
    """

    assert metric in ['discounted', 'average', 'cumulative'], \
        "Unsupported metric"

    old_epsilon = policy.get_epsilon()
    policy.set_epsilon(eval_epsilon)
    old_clip = mdp.clip_reward
    mdp.clip_reward = clip

    out = Parallel(n_jobs=n_jobs)(
        delayed(_eval)(
            mdp, policy, metric=metric, video=video,
            save_video=save_video, save_path=save_path,
            append_filename=('_%s' % append_filename).rstrip('_') + '_%s' % eid,
            initial_actions=initial_actions
        )
        for eid in range(n_episodes)
    )

    policy.set_epsilon(old_epsilon)
    mdp.clip_reward = old_clip

    values, steps = np.array(zip(*out))
    return values.mean(), values.max(), 2 * values.std() / np.sqrt(n_episodes), \
           steps.mean(), steps.max(), 2 * steps.std() / np.sqrt(n_episodes)


def _eval(mdp, policy, metric='cumulative', video=False, save_video=False,
          save_path='', append_filename='', initial_actions=None):
    frames = []
    gamma = mdp.gamma if metric == 'discounted' else 1
    ep_performance = 0.0
    df = 1.0  # Discount factor
    frame_counter = 0

    # Get current state
    state = mdp.reset()

    # Force start
    if initial_actions is not None:
        state, _, _, info = mdp.step(np.random.choice(initial_actions))
        lives_count = info['ale.lives']

    if save_video:
        frames.append(state[-1])

    reward = 0
    done = False

    # Start episode
    while not done:
        frame_counter += 1

        if initial_actions is not None:
            if info['ale.lives'] < lives_count:
                lives_count = info['ale.lives']
                state, _, _, _ = mdp.step(np.random.choice(initial_actions))

        # Select and execute the action, get next state and reward
        action = policy.draw_action(np.expand_dims(state, 0), done, evaluation=True)
        action = int(action)
        next_state, reward, done, info = mdp.step(action)

        # Update figures of merit
        ep_performance += df * reward  # Update performance
        df *= gamma  # Update discount factor

        # Render environment
        if video:
            mdp.render(mode='human')

        # Update state
        state = next_state
        if save_video:
            frames.append(state[-1])

    if metric == 'average':
        ep_performance /= frame_counter

    if save_video:
        append_filename = append_filename.lstrip('_')
        filename = save_path + 'eval_%s_score_%s_steps_%s.gif' % \
                               (append_filename, ep_performance, frame_counter)
        filename = time.strftime(filename)
        imageio.mimsave(filename, frames)

    return ep_performance, frame_counter
