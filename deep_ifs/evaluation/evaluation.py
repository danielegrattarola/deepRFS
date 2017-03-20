from joblib import Parallel, delayed
import numpy as np


def evaluate_policy(mdp, policy, metric='cumulative', n_episodes=1,
                    max_ep_len=np.inf, video=False, n_jobs=1):
    """
        This function evaluate a policy on the given environment w.r.t.
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
            max_ep_len (int, inf): allow evaluation episodes to run at most
                this number of frames.
            video (bool, False): whether to render the environment.
            n_jobs (int, 1): the number of processes to use for evaluation
                (leave default value if the feature extraction model runs
                on GPU).
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
    out = Parallel(n_jobs=n_jobs)(
        delayed(_eval)(
            mdp, policy, metric=metric, max_ep_len=max_ep_len, video=video
        )
        for _ in range(n_episodes)
    )

    values, steps = np.array(zip(*out))
    return values.mean(), 2 * values.std() / np.sqrt(n_episodes), \
           steps.mean(), 2 * steps.std() / np.sqrt(n_episodes)


def _eval(mdp, policy, metric='cumulative', max_ep_len=np.inf, video=False):
    gamma = mdp.gamma if metric == 'discounted' else 1
    ep_performance = 0.0
    df = 1.0  # Discount factor
    frame_counter = 0

    # Get current state
    state = mdp.reset()
    reward = 0
    done = False

    # Start episode
    while not done and frame_counter <= max_ep_len:
        frame_counter += 1

        # Select and execute the action, get next state and reward
        action = policy.draw_action(np.expand_dims(state, 0), done,
                                    evaluation=True)
        next_state, reward, done, info = mdp.step(action)

        # Update figures of merit
        ep_performance += df * reward  # Update performance
        df *= gamma  # Update discount factor

        # Render environment
        if video:
            mdp.render(mode='human')

        # Update state
        state = next_state

    if metric == 'average':
        ep_performance /= frame_counter

    return ep_performance, frame_counter
