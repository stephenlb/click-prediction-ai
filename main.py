from dataclasses import dataclass
from typing import override
import pygame
import sys
from abc import abstractmethod, ABC, ABCMeta

import torch
from torch import nn
from torchvision.transforms import v2

def fade_color(color, factor):
    return [max(0, int(c * factor)) for c in color]

@dataclass
class Window:
    width: int
    height: int
    screen: pygame.Surface | None = None

@dataclass
class Game:
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
                print(
                    event.dict,
                    event.pos
                )
            self.push([30, event.pos, [
                [[80,120, 80], 20],
                [[160, 240, 160], 10]
            ]])
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
    print(numberCoords)
    if numberCoords < 8:
        return

    features = clicks[-8:-2]
    label = clicks[-2]
    output = predictor([features])

class NNPredictor(nn.Module):
    def __init__(self, game):
        super().__init__()
        ## TODO use GPU??!?!?!?!?
        ## TODO use GPU??!?!?!?!?
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        ## TODO use GPU??!?!?!?!?
        ## TODO use GPU??!?!?!?!?
        print(f"Accelerator devices is: {self.device}")
        print(f"Game Width and Height{[game.window.width, game.window.height]}")
        self.clicks = 0
        self.register_buffer(
            'normalization',
             torch.as_tensor([game.window.width, game.window.height])
        )
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
            torch.nn.Tanh(),
        )
        ## TODO DO THE DECODER
        ## TODO DO THE DECODER
        self.decoder = None
        ## TODO DO THE DECODER
        ## TODO DO THE DECODER
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3),
        self.loss = torch.nn.MSELoss(),

    def encoder(self, features):
        features = torch.as_tensor(features, dtype=torch.float32)
        features = features.reshape(-1, 2)
        features = features / self.normalization
        features = features.reshape(-1)
        return features

    def decode(self, output):
        output = output.reshape(-1, 2)
        output = output * self.normalization
        output = output.reshape(-1)
        return output

    def forward(self, features):
        ## Encoding Step
        print(f"BEFORE: {features}")
        features = self.encoder(features)
        print(f"AFTER: {features}")
        output = self.model(features)
        print(f"OUTPUT: {output}")
        decoded = self.decode(output)
        print(f"DECODED: {decoded}")
        return decoded
        #output = self.model(features)
        ## TODO
        ## TODO  decode = output.....
        ## TODO
        ## TODO
        #return output

    def train(self, features, labels):
        pass
        
        

def main():
    fps = 60
    pygame.init()
    marker = PointMarker()
    game = Game(
        draw=pygame.draw,
        window=Window(
            width=900,
            height=500
        ),
        clock=pygame.time.Clock()
    )
    game.window.screen = pygame.display.set_mode(
        (game.window.width, game.window.height)
    )
    predictor = NNPredictor(game)

    while True:
        game.window.screen.fill(
            (40, 40, 60)
        )

        for event in pygame.event.get():
            marker.event_hook(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        game.draw.circle(
            game.window.screen,
            (240, 120, 120),
            (game.window.width / 2, game.window.height / 2),
            12
        )
        marker.draw(game)
        predictions(predictor, marker, game)
        pygame.display.flip()
        game.clock.tick(fps)

if __name__ == "__main__":
    main()
