import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
#from tf_agent.replay_buffers import py_uniform_replay_buffer
import random as rand
import numpy as np
from numpy.random import randint, uniform

np_config.enable_numpy_behavior()
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()

# build the dqn model
def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx+nu,1))                        # input
    state_out1 = layers.Dense(16, activation="relu")(inputs)      # hidden layer 1
    state_out2 = layers.Dense(32, activation="relu")(state_out1)  # hidden layer 2
    state_out3 = layers.Dense(64, activation="relu")(state_out2)  # hidden layer 3
    state_out4 = layers.Dense(64, activation="relu")(state_out3)  # hidden layer 4
    outputs = layers.Dense(1)(state_out4)                         # output

    model = tf.keras.Model(inputs, outputs)                       # create the NN
    
    return model

def update(xu_batch, cost_batch, xu_next_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = Q_target(xu_next_batch, training=True)   
        # Compute 1-step targets for the critic loss
        y = cost_batch + DISCOUNT*target_values                            
        # Compute batch of Values associated to the sampled batch of states
        Q_value = Q(xu_batch, training=True)                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))

# epsilon-greedy
def get_action(exploration_prob, nu, Q, x):
    # with probability exploration_prob take a random control input
    if(uniform() < exploration_prob):
        u = randint(0, nu)
    # otherwise take a greedy control
    else:
       action_values = Q.predict(x)
       best_action_index = tf.argmin(action_values)
       u = tf2np(action_values[best_action_index])
    return u


def dqn_learning(env, gamma, Q, Q_target, nEpisodes, maxEpisodeLength, \
               learningRate, exploration_prob, exploration_decreasing_decay, \
               min_exploration_prob, compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' 
        DQN learning algorithm:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # replay buffer
    replay_buffer = []
    # Keep track of the cost-to-go history (for plot)
    h_ctg = []
    # Make a copy of the initial Q table guess
    Q = tf.keras.models.clone_model(Q)

    capacity_buffer = 1000
    batch_size = 32
    c_step = 4
    ep = 0

    # for every episode
    for i in range(nEpisodes):
        # reset the state
        env.reset()
        J = 0
        ep += 1
        gamma_to_the_i = 1
        # simulate the system for maxEpisodeLength steps
        for k in range(int(maxEpisodeLength)):
            # useful variable
            x = env.x
      
            # epsilon-greedy action selection
            u = get_action(exploration_prob, env.nu, Q, x)
            
            # observe cost and next state (step = calculate dynamics)
            x_next, cost = env.step(u)
            u_next = get_action(exploration_prob, env.nu, Q, x_next)
        
            ###### if there are no 32 elements in the batch we cannot extract anything

            # store the experience (s,a,r,s') in the replay_buffer
            experience = [x, u, cost, x_next, u_next]
            replay_buffer.append(experience)

            # check the length of the replay_buffer and resize it if it's bigger than capacity_buffer
            del replay_buffer[:-capacity_buffer]
            
            # Randomly sample minibatch (size of batch_size) of experience from replay_buffer
            batch = rand.choices(replay_buffer, k=batch_size)
            x_batch, u_batch, cost_batch, x_next_batch, u_next_batch = list(zip(*batch))
            
            x_batch = np.concatenate(x_batch, axis=1)
            u_batch = np.asarray(u_batch)
            cost_batch = np.asarray(cost_batch)
            xu_batch = np.append(x_batch, u_batch)
            x_next_batch = np.concatenate(x_next_batch, axis=1)
            u_next_batch = np.asarray(u_next_batch)
            xu_next_batch = np.append(x_next_batch, u_next_batch)
           
            # convert numpy to tensorflow
            xu_batch = np2tf(xu_batch) 
            cost_batch = np2tf(cost_batch)
            xu_next_batch = np2tf(xu_next_batch)

            # update weights
            update(xu_batch, cost_batch, xu_next_batch)
            
            # keep track of the cost to go
            J += gamma_to_the_i * cost
            gamma_to_the_i *= gamma

            # Periodically update target network (period = c_step)
            if k % c_step == 0:
                Q_target.set_weights(Q.get_weights())
        
        J_avg = J / ep

        h_ctg.append(J_avg)
        # update the exploration probability with an exponential decay: 
        # eps = exp(-decay*episode)
        exploration_prob = max(np.exp(-exploration_decreasing_decay*ep), min_exploration_prob)
        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(k%nprint==0):
            print("Q learning - Iter %d, J=%.1f, eps=%.1f"%(k,J,100*exploration_prob))
            if(plot):
                V, pi = compute_V_pi_from_Q(env, Q)
                env.plot_V_table(V)
                env.plot_policy(pi)
    
    return Q, h_ctg

nx = 2
nu = 1
QVALUE_LEARNING_RATE = 1e-3
DISCOUNT = 0.99

# Create critic and target NNs
Q = get_critic(nx, nu)
Q_target = get_critic(nx, nu)

Q.summary()

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

q_net = Sequential()
q_net.add(Dense(64, input_dim=4, activation='relu',kernel_initializer='he_uniform'))
q_net.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
q_net.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))
#q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.mean_squared_error)

q_net.summary()

# Set initial weights of targets equal to those of the critic
Q_target.set_weights(Q.get_weights())

# Set optimizer specifying the learning rates
critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

w = Q.get_weights()
for i in range(len(w)):
    print("Shape Q weights layer", i, w[i].shape)
    
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))
    
print("\nDouble the weights")
for i in range(len(w)):
    w[i] *= 2
Q.set_weights(w)

w = Q.get_weights()
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))

print("\nSave NN weights to file (in HDF5)")
Q.save_weights("namefile.h5")

#update()

print("Load NN weights from file\n")
Q_target.load_weights("namefile.h5")

w = Q_target.get_weights()
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))