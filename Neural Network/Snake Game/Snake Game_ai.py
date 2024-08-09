import pygame
import time
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

pygame.init()

dis_width = 800
dis_height = 600
snake_block = 10
snake_speed = 30

white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game')

clock = pygame.time.Clock()

snake_block = 10
snake_speed = 120

font_style = pygame.font.SysFont(None, 50)
score_font = pygame.font.SysFont(None, 35)


def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])


model = Sequential()
model.add(Dense(24, input_shape=(9,), activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(3, activation='softmax'))
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer)


def get_state(snake_head, food_pos, snake_list):

    dx = food_pos[0] - snake_head[0]
    dy = food_pos[1] - snake_head[1]

    obstacles = [
        (snake_head[0] - snake_block, snake_head[1]),
        (snake_head[0] + snake_block, snake_head[1]),
        (snake_head[0], snake_head[1] - snake_block),
        (snake_head[0], snake_head[1] + snake_block)
    ]
    for obstacle in obstacles:
        if obstacle in snake_list:
            obstacles[obstacles.index(obstacle)] = 1
        else:
            obstacles[obstacles.index(obstacle)] = 0
    
    state = np.array([dx, dy] + obstacles[:-1])
    return state


def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        
        action = random.randint(0, 2)
    else:
       
        act_values = model.predict(state.reshape(1, 9))
        action = np.argmax(act_values[0])
    return action


def game_loop():
    game_over = False
    x1 = dis_width / 2
    y1 = dis_height / 2
    x1_change = 0
    y1_change = 0
    snake_List = []
    Length_of_snake = 1
    foodx = round(random.randrange(0, dis_width * 2 - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height * 2 - snake_block) / 10.0) * 10.0
    
    
    epsilon = 1.0
    gamma = 0.9
    memory = []
    batch_size = 32
    start_time = time.time()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        state = get_state([x1, y1], [foodx, foody], snake_List)
        action = select_action(state, epsilon)

        if action == 0:
            x1_change = -snake_block
            y1_change = 0
        elif action == 1:
            x1_change = snake_block
            y1_change = 0
        else:
            x1_change = 0
            y1_change = -snake_block
            
        if x1 == foodx and y1 == foody:
         foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
         foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
         Length_of_snake += 1

        x1 += x1_change
        y1 += y1_change

        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        our_snake(snake_block, snake_List)

        pygame.display.update()

        if x1 >= dis_width:
         x1 = 0
        elif x1 < 0:
         x1 = dis_width - snake_block
        elif y1 >= dis_height:
         y1 = 0
        elif y1 < 0:
         y1 = dis_height - snake_block
         
        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1

        score = Length_of_snake - 1

        clock.tick(snake_speed)

    pygame.quit()


game_loop()
