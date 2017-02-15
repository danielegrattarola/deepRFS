def collect_sars(env, policy):
    pass

def balance_dataset(sars):
    pass

def build_farf(nn, sars):
    # Build FARF' dataset using SARS' dataset:
        # F = NN[0].features(S)
        # A = A
        # R = R
        # F' = NN[0].features(S')
    pass

def build_sfadf(nn_stack, nn, sars):
    # Build SFADF' dataset using SARS' dataset:
        # S = S
        # F = NN_stack.s_features(S)
        # A = A
        # D = NN[i-1].features(S) - NN[i-1].features(S')
        # F' = NN_stack.s_features(S')
    pass

def build_sares(model, sfadf):
    # Build SARes dataset from SFADF':
        # S = S
        # A = A
        # Res = D - M(F)
    pass

def build_fadf(nn_stack, nn, sars, sfadf):
    # Build new FADF' dataset from SARS' and SFADF':
        # F = NN_stack.s_features(S) + NN[i].features(S)
        # A = A
        # D = SFADF'.D
        # F' = NN_stack.s_features(S') + NN[i].features(S')
    pass

def build_global_farf(nn_stack, sars):
    # Build FARF' dataset using SARS' dataset:
        # F = NN_stack.s_features(S)
        # A = A
        # R = R
        # F' = NN_stack.s_features(S')
    pass
