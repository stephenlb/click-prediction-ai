from dataclasses import dataclass
from typing import override
import pygame
import sys
from abc import abstractmethod, ABC, ABCMeta
import json

import torch
from torch import nn
from torchvision.transforms import v2

### TODO more than one training epoch
### TODO 
### TODO 
### TODO 
### TODO 
### 
### TODO add random clicks for lots of training data (and learning rate)
### TODO hot spots (heatmap) improve make prettier
### TODO 
### TODO IMPROVE MODEL
### TODO LSTM????
### TODO imporove model for better accuracy
### TODO reduce params
### TODO increase number of input xy coords
### TODO 
### 


def fade_color(color, factor):
    return [max(0, int(c * factor)) for c in color]

@dataclass
class Window:
    width: int
    height: int
    screen: pygame.Surface | None = None

@dataclass
class Game:
    ai: list
    ai_history: list[list]
    heatmap: pygame.Surface
    draw: pygame.draw
    window: Window
    clock: pygame.time.Clock

class Drawable(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def event_hook(cls, event: pygame.event.Event):
        pass

    @classmethod
    @abstractmethod
    def draw(cls, game: Game):
        pass

class PointMarker(Drawable):
    def __init__(self):
        super().__init__()
        self.point_list: list[list] = []
        self.ai_point_list: list[list] = []
        self.previous_clicks: list[list] = []
        self.previous_clicks_position: int = 0

    def add(self, point: list):
        self.push([30, (point[0],point[1]), [
            [[80,120, 80], 20],
            [[160, 240, 160], 10]
        ]])
        
    def push(self, point: list):
        timer, pos, state = point
        x = pos[0]
        y = pos[1]
        self.point_list.append(point)
        self.ai_point_list.append(x)
        self.ai_point_list.append(y)

    @override
    def event_hook(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == pygame.BUTTON_LEFT:
                self.previous_clicks.append([event.pos[0], event.pos[1]])
                print(event.dict, event.pos)
                self.add(event.pos)
    @override
    def draw(self, game: Game):
        for circle in self.point_list[:]:
            timer, pos, state = circle
            game.draw.circle(game.window.screen, state[0][0], pos, 20)
            game.draw.circle(game.window.screen, state[1][0], pos, 10)
            circle[0] -= 1
            if circle[0] % 10 == 0:
                state[0][0] = fade_color(state[0][0], 0.7)
                state[1][0] = fade_color(state[1][0], 0.5)

            if circle[0] <= 0:
                self.point_list.remove(circle)

def predictions(predictor, marker, game):
    clicks = marker.ai_point_list
    numberCoords = len(clicks)

    ## Only run AI when user clicks
    if predictor.clicks >= numberCoords:
        return

    ## Set number of clicks so we don't run next frame again
    predictor.clicks = numberCoords

    ## Only allow AI to run if we have enough data samples
    if numberCoords < predictor.inputLength + 2:
        return

    ## TODO add more xy coords
    ## TODO add more xy coords
    features = clicks[-predictor.inputLength - 2:-2]
    label = clicks[-2:]
    prediction = predictor.train(features, label)
    print(f"prediction:{prediction}")

class NNPredictor(nn.Module):
    def __init__(self, game):
        super(NNPredictor, self).__init__()
        self.inputLength = 6
        self.game = game
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Accelerator devices is: {self.device}")
        print(f"Game Width and Height{[game.window.width, game.window.height]}")
        self.clicks = 0
        self.register_buffer(
            'normalization',
             torch.as_tensor([game.window.width, game.window.height])
        )
        ## Neva said to use CNN + also maybe LSTM in front?!?!?!
        ## Neva said LSTM first, give it 3 guesses instaed of 1
        self.model = torch.nn.Sequential(
            ## TODO change input for CNN
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(105600, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        #self.model = torch.nn.Sequential(
        #    nn.Linear(6, 8),
        #    nn.ReLU(),
        #    nn.Linear(8, 8),
        #    nn.ReLU(),
        #    nn.Linear(8, 2),
        #    nn.Sigmoid(),
        #)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-1)
        #self.optimizer.param_groups[0]['weight_decay'] = 0.1
        #self.loss = torch.nn.MSELoss()
        #self.loss = torch.nn.L1Loss()
        self.loss = self.loss
        self.losses = []

    def loss(self, output, labels):
        print("output",output)
        print("labels",labels)
        return torch.mean(torch.abs(output - labels).pow(0.5))
        #return ((labels - output) ** 2).mean()

    ## TODO render the output
    ## TODO render the output
    ## TODO render the output
    ## TODO render the output
    def encoderCNN(self, coords, game):
        ## TODO can be smaller input
        width = game.window.width
        height = game.window.height
        output = torch.zeros(1, 1, width, height) 
        for pos in range(0, len(coords), 2):
            x = coords[pos]
            y = coords[pos+1]
            output[0, 0, x, y] = 1.0
        return output

    def encoder(self, features):
        features = torch.as_tensor(features, dtype=torch.float32)
        features = features.reshape(-1, 2)
        features = features / self.normalization
        features = features.reshape(-1)
        return features

    def encoderCNNold(self, features):
        print('CNN ENCODER: ', features)
        #matrix = features.detach().cpu().numpy()
        matrix = features.reshape(1,5,5)
        print('CNN MATRIX: ', matrix)
        #matrix = torch.as_tensor([torch.as_tensor(matrix) for feature in features])
        #matrix = [[matrix] for feature in features]
        ## TODO if we use numpin impott numpy
        #matrix = numpy.array(matrix)
        #matrix = torch.as_tensor(matrix)
        return matrix
        
    def decoder(self, output):
        output = output.reshape(-1, 2)
        output = output * self.normalization
        output = output.reshape(-1)
        return output

    ## RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d,
    ## but got input of size: [6]

    def forward(self, features):
        #features = self.encoder(features)
        print('features: ', features)
        features = self.encoderCNN(features, self.game)
        print('features Post encoderCNN: ', features)
        #features = self.encoderCNN(features)
        output = self.model(features)
        decoded = self.decoder(output)
        return decoded

    def train(self, features, labels):
        output = self.forward(features)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        loss = self.loss(output, labels)
        self.losses.append(loss)
        cost = sum(self.losses) / len(self.losses)
        print(f"loss: {loss}")
        print(f"cost: {cost}")

        coords = output.detach().cpu().numpy()
        print(f"coords: {(coords[0], coords[1])}")

        ## Set AI Prediction Pixel XY Coords
        self.game.ai = [int(coords[0]), int(coords[1])]
        self.game.ai_history.append([int(coords[0]), int(coords[1])])

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return output
        
def draw_heatmap(game):
    ## TODO remove ai_history
    coords = game.ai
    size = (game.window.width, game.window.height)
    surface = pygame.Surface(size, pygame.SRCALPHA) 
    surface.fill((255, 255, 255, 255))
    for i in range(4):
        game.draw.circle(
            surface,
            (255,253,253),
            #(255 - 10 * i, 255 -  10 * i, 255 -  10* i, 255 -  10* i),
            coords,
            (4 * i)
        )
        game.heatmap.blit(surface, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    game.window.screen.blit(game.heatmap, (0,0))#, special_flags=pygame.BLEND_RGB_MULT)

def main():
    frame = 0
    fps = 60
    pygame.init()
    marker = PointMarker()

    ## Load Previous Clicks
    try:
        with open("clicks.json", "r") as file:
            marker.previous_clicks = json.load(file)
        print(f"Previous Clicks: {marker.previous_clicks}")
    except:
        print(f"No previous click file loaded.")

    previous_click_count = len(marker.previous_clicks)

    size = (900, 500)
    game = Game(
        ai=[450, 250], ## coordiantes for the AI Prediction
        ai_history=[],
        heatmap=pygame.Surface(size, pygame.SRCALPHA),
        draw=pygame.draw,
        window=Window(width=size[0], height=size[1]),
        clock=pygame.time.Clock()
    )
    game.heatmap.fill((255, 255, 255))
    game.window.screen = pygame.display.set_mode(size, pygame.SRCALPHA)
    predictor = NNPredictor(game)
    background = pygame.Surface(size)
    background.fill((100, 150, 200))

    while True:
        game.window.screen.fill((255,255,255,255))
        for event in pygame.event.get():
            marker.event_hook(event)
            if event.type == pygame.QUIT:
                with open("clicks.json", "w") as file:
                    json.dump(marker.previous_clicks, file)
                pygame.quit()
                sys.exit()

        if frame % 7 == 0 and \
        previous_click_count > marker.previous_clicks_position:
            click = marker.previous_clicks[marker.previous_clicks_position]
            marker.previous_clicks_position += 1
            marker.add((click[0], click[1]))

        game.window.screen.blit(background, (0, 0))
        draw_heatmap(game)
        game.draw.circle(
            game.window.screen,
            (240, 220, 120),
            (game.ai[0], game.ai[1]),
            20
        )
        marker.draw(game)
        predictions(predictor, marker, game)
        pygame.display.flip()
        game.clock.tick(fps)
        frame += 1

if __name__ == "__main__":
    main()
