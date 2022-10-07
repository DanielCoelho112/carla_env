from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model

# we will train and predict in different threats
IM_WIDTH = 640
IM_HEIGHT = 480
REPLAY_MEMORY_SIZE = 5_000 # 5000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = 'Xception'
MEMORY_FRACTION = 0.8 # how much gpu to use. Not needed.
MIN_REWARD = -200
DISCOUNT = 0.99
EPISODES = 100
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10

class DQNAgent():
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()
        
        self.terminate = False 
        self.last_logged_episode = 0
        self.training_initialized = False 
    
    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))
        # usually, when we choose a model like this, we remove the first and last layer
        
        x = base_model.output  
        x = GlobalAveragePooling2D()(x)
        
        predictions = Dense(3, activation='linear')(x)
        model = Model(inputs= base_model.input, outputs=predictions)
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        return model 
    
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)
    
    def train(self):
        pass