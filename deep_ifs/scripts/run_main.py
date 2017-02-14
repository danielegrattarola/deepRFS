# Algorithm pseudocode

# NN[i]: is the i-th neural network and provides the features(S) method that
#	returns all 512 features produced when state S is given as input to the network

# NN_stack: contains all trained neural networks so far, provides the s_features(S)
# 	method that returns all **selected** features produced when state S is given as
#	input to each network.

# Policy = fully random

# Main loop:

    # Collect SARS' samples with policy
    # Fix dataset to account for imbalance (proportional to the number of transitions for each reward class)

    # Fit neural network NN[0]: S -> R, using SARS' dataset

    # Build FARF' dataset using SARS' dataset:
        # F = NN[0].features(S)
        # F' = NN[0].features(S')
    # Select support features of NN[0] with IFS using FARF' dataset (target = R)

    # For i in range(1, N):
        # Build SFADF' dataset using SARS' dataset:
            # S = S
            # F = NN_stack.s_features(S)
            # A = A
            # D = NN[i-1].features(S) - NN[i-1].features(S')
            # F' = NN_stack.s_features(S')

        # Fit model M: F -> D, using SFADF' dataset

        # Build SARes dataset from SFADF':
            # S = S
            # A = A
            # Res = D - M(F)
        # Fit neural network NNi: S -> Res, using SARes dataset

        # Build new FADF' dataset from SARS' and SFADF':
            # F = NN_stack.s_features(S) + NN[i].features(S)
            # A = A
            # D = SFADF'.D
            # F' = NN_stack.s_features(S') + NN[i].features(S')
        # Select support features of NNi with IFS using new FADF' dataset

        # If (no new feature is selected) or (R2 of D is below a threshold):
            # Break

    # Update policy with FQI (using support features of all steps), decrease randomicity

epsilon = 1.0
policy = epsilon_FQI(estimator, state_dim, action_dim, discrete_action, epsilon, gamma, horizon)

for i in range(iterations):
    sars = collect_sars(policy)  # State, action, reward, next_state
    sars = balance_dataset(sars)  # Either this, or just assign different weights to positive classes in nn.fit

    nn_stack = NNstack()  # To store all neural networks and IFS support

    nn = ConvNet(image_shape, reward_dim)  # Maps frames to reward
    nn.fit(sars.s, sars.r)

    farf = build_farf(nn, sars)  # Features, action, reward, next_features
    ifs = IFS(**rfs_params)
    ifs.fit(split_dataset(farf, feature_dim, action_dim, reward_dim))
    support = ifs.get_support()

    nn_stack.add(0, nn, support)

    for j in range(1, depth):
        previous_support = nn_stack.get_support()

        sfadf = build_sfadf(nn_stack, nn, sars)  # State, all features, action, dynamics, all next_features
        model = ExtraTreeRegressor()
        model.fit(sfadf.f, sfadf.d)

        sares = build_sares(model, sfadf)  # Res = D - model(F)

        nn = ConvNet(image_shape, res_size, support_dim)  # Maps frames to residual dynamics
        nn.fit(sares.s, sares.res)

        fadf = build_fadf(nn_stack, nn, sars, sfadf)  #  All features, action, dynamics, all next_features
        ifs = IFS(**rfs_params)
        ifs.fit(split_dataset(fadf, feature_dim, action_dim, dynamics_dim))
        support = ifs.get_support()

        nn_stack.add(j, nn, support)

        if not nn_stack.support_changed(previous_support) or ifs.scores_confidences_ < threshold:
            print 'Done.'
            break

    global_farf = build_global_farf(nn_stack, sars)  # All features, action, reward, all next_features
    all_features_dim = nn_stack.get_support_dim()
    policy.fit_on_dataset(split_dataset_for_fqi(global_farf), all_features_dim)  # Need to pass new dimension of "states" to instantiate new FQI
    policy.epsilon_step(epsilon_rate)

