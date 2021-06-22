#%%
import numpy as np
import pygame
from copy import deepcopy
from time import sleep
import DQN
from tqdm import tqdm

#pygame.init()
#pygame.display.set_caption('snake game')

ACTION_SPACE_SIZE = 4
EPISODES = 2000
epsilon = 1

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50


class Terrain:

    def __init__(self,nWidth=21,nHeight=11):
        
        self.nWidth = nWidth
        self.nHeight = nHeight
        self.grid = [[0 for i in range(self.nWidth)] for _ in range(self.nHeight)]
        #self.appleLocation = (31,40)
        self.appleLocation = (np.random.randint(self.nHeight),np.random.randint(self.nWidth))
        self.grid[self.appleLocation[0]][self.appleLocation[1]] = 2
        
    def getnWidth(self):
        return self.nWidth

    def getnHeight(self):
        return self.nHeight

    def getgrid(self):
        return self.grid


    def getAppleLocation(self):
        return self.appleLocation

    def generateNewApple(self):
        self.appleLocation = (np.random.randint(self.nHeight),np.random.randint(self.nWidth))


class Snake:

    def __init__(self):

        self.snakeList = []
        self.snakeLength = 0
        self.direction = 0

    def getSnakeList(self):
        return self.snakeList

    def getSnakeLength(self):
        return self.snakeLength

    def getDirection(self):
        return self.direction
    
    def setDirection(self,new_direction):
        self.direction = new_direction

    def setSnakeLength(self,new_length):
        self.snakeLength = new_length

    def setSnakeList(self,new_L):
        self.snakeList = new_L
        self.setSnakeLength(len(new_L))
    
    def enlargeSnake(self):
        self.setSnakeLength(self.getSnakeLength()+1)


class Game:

    def __init__(self):

        self.field = Terrain()
        self.snake = Snake()
        self.width = self.field.getnWidth()*10
        self.height = self.field.getnHeight()*10
        #self.board = pygame.display.set_mode((self.width,self.height))
        self.snake_block = 10
        self.snake_speed = 30
        self.clock = pygame.time.Clock()

        snake_ini = [(self.field.getnHeight()//2,self.field.getnWidth()//2)]
        self.snake.setSnakeList(snake_ini)
        self.gameOver = False

    def updateGrid(self,newSnakeList):

        if self.field.grid[newSnakeList[0][0]][newSnakeList[0][1]]==2:
            #print("apple caught")
            self.snake.enlargeSnake() #plus grand score et plus grand serpent et pleins de truc compliqués
            self.field.generateNewApple()
            while self.field.getAppleLocation() in newSnakeList:
                self.field.generateNewApple()
            self.field.grid[self.field.getAppleLocation()[0]][self.field.getAppleLocation()[1]]=2
            reward = 100
            #print(f"snake len {self.snake.getSnakeLength()}")
        
        if len(newSnakeList) > self.snake.getSnakeLength():
            old_tail = newSnakeList.pop(-1)
            self.field.grid[old_tail[0]][old_tail[1]] = 0
            reward = -1
        for x in newSnakeList:
            self.field.grid[x[0]][x[1]] = 1
        
        
        self.snake.setSnakeList(newSnakeList)

        return reward


    """ def render(self):

        #pygame.display.update()

        black = (0,0,0)
        white = (255,255,255)
        red = (255,0,0)
        blue = (0,0,255)


        self.board.fill(black)
        for x in self.snake.getSnakeList():
            pygame.draw.rect(self.board,white,[x[1]*10,x[0]*10,self.snake_block,self.snake_block])
            pygame.display.update()

        pygame.draw.rect(self.board,red,[self.field.getAppleLocation()[1]*10,self.field.getAppleLocation()[0]*10,self.snake_block,self.snake_block])
        #self.clock.tick(self.snake_speed)

        pygame.display.update() """

    def turn(self,action):

        """ for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit() """
        #action = np.random.randint(4)
        #action = 3

        if action == 0:
            dx = -1
            dy = 0
        elif action == 1:
            dx = 0
            dy = -1
        elif action == 2:
            dx = 1
            dy = 0
        elif action == 3:
            dx = 0
            dy = 1

        new_head = (self.snake.getSnakeList()[0][0]+dy,self.snake.getSnakeList()[0][1]+dx)

        if new_head[0]<0 or new_head[0]>=self.field.getnHeight() or new_head[1]<0 or new_head[1]>=self.field.getnWidth():
            self.gameOver = True
            reward = -100
            #print("You went out of the bounds")
        elif new_head in self.snake.getSnakeList():
            self.gameOver = True
            reward = -100
            #print("You crashed into your own body")
        else:
            newSnakeList = deepcopy(self.snake.getSnakeList())
            newSnakeList.insert(0,new_head)

            #print(f'snake list {self.snake.getSnakeList()}')
            #print(f"new snakelist {newSnakeList}")
            #print(f"apple position {self.field.getAppleLocation()}")
            reward = self.updateGrid(newSnakeList)
            #self.render()

        return self.field.grid,reward,self.gameOver


game = Game()


ep_rewards = [-200]

## For more repetitive results
#random.seed(1)
#np.random.seed(1)
#tf.set_random_seed(1)
#
# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

agent = DQN.DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    game = Game()
    current_state = game.field.getgrid()

    while not game.gameOver:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, ACTION_SPACE_SIZE)

        new_state, reward, done = game.turn(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        #if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            #env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1


        # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        #if average_reward >= MIN_REWARD:
        #    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    



# %%
#To do 
#wall colision
#snake colliding himself
#Q learning implementation


