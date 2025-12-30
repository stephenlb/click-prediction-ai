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


class DebugShape(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def _prefix(self):
        return f"[DebugShape{'' if self.name is None else ':'+self.name}]"

    def forward(self, x):
        p = self._prefix()
        if isinstance(x, torch.Tensor):
            print(p, "shape =", tuple(x.shape))
        elif isinstance(x, (list, tuple)):
            print(p, "list/tuple shapes =", [tuple(t.shape) for t in x])
        elif isinstance(x, dict):
            print(p, "dict shapes =", {k: tuple(v.shape) for k,v in x.items()})
        else:
            print(p, "unsupported type:", type(x))
        return x


class DebugValue(nn.Module):
    def __init__(self, name=None, max_elements=20):
        super().__init__()
        self.name = name
        self.max_elements = max_elements

    def _prefix(self):
        return f"[DebugValue{'' if self.name is None else ':'+self.name}]"

    def _print_tensor(self, t):
        p = self._prefix()
        flat = t.flatten()
        if flat.numel() <= self.max_elements:
            print(p, "value =", t)
        else:
            shown = flat[:self.max_elements]
            print(p, f"value (first {self.max_elements}/{flat.numel()}) =", shown)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            self._print_tensor(x)
        elif isinstance(x, (list, tuple)):
            for i, t in enumerate(x):
                print(self._prefix(), f"[{i}]")
                self._print_tensor(t)
        elif isinstance(x, dict):
            for k, t in x.items():
                print(self._prefix(), f"[{k}]")
                self._print_tensor(t)
        else:
            print(self._prefix(), "unsupported type:", type(x))
        return x


class ContextConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding="same")
        # self.act = nn.LeakyReLU(0.01)
        self.act = nn.ReLU()
        # self.act = nn.LogSoftmax()#smooth is nicer for more placticity


    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)

        # channel pooling
        maxvals = y.max(dim=1, keepdim=True).values    # (B,1,H,W)
        first = y[:, 0:1, :, :] + maxvals             # updated channel 0
        rest  = y[:, 1:, :, :]                        # untouched channels

        out = torch.cat([first, rest], dim=1)         # (B,C,H,W)
        return out


class SampleSpace(nn.Module):
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device

        # spatial softmax
        probs = torch.softmax(x.view(B, C, -1), dim=-1).view(B, C, H, W)

        # coordinate grids
        ys = torch.arange(H, device=device).float() / (H - 1)
        xs = torch.arange(W, device=device).float() / (W - 1)

        grid_x = xs.view(1, 1, 1, W).expand(B, C, H, W)
        grid_y = ys.view(1, 1, H, 1).expand(B, C, H, W)

        ex = (probs * grid_x).sum(dim=(2, 3))  # expected x in [0,1]
        ey = (probs * grid_y).sum(dim=(2, 3))  # expected y in [0,1]

        return torch.stack([ex, ey], dim=-1)    # (B, C, 2)

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
             torch.as_tensor([game.window.width, game.window.height], dtype=torch.float32)
        )
        ## Neva said to use CNN + also maybe LSTM in front?!?!?!
        ## Neva said LSTM first, give it 3 guesses instaed of 1
        self.model = torch.nn.Sequential(
            ## TODO change input for CNN
            ContextConv(1,5,15),
            # ContextConv(5,5,4),
            ContextConv(5,8,15),
            SampleSpace(),
            nn.Flatten(),
            nn.Linear(16,32),
            # nn.Dropout(0.2),
            nn.Softmin(),#avoids weird shooting behiviors
            nn.Linear(32,32),
            nn.Softmin(),
            nn.Linear(32,2),
            nn.Sigmoid(),
        )

        self.to(self.device)
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

    ## RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d,
    ## but got input of size: [6]
    def forward(self, features):
        #features = self.encoder(features)
        print('features: ', features)
        features = self.encoderCNN(features, self.game)
        print('features Post encoderCNN: ', features)
        output = self.model(features)
        decoded = self.decoder(output)
        return decoded

    def loss(self, output, labels):
        print("output",output)
        # print("labels",labels)
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
        # Encode as [batch, x, y, channel] so channels stay last while marking click positions
        output = torch.zeros(1,1, width, height,device=self.device)
        for pos in range(0, len(coords), 2):
            x = coords[pos]
            y = coords[pos+1]
            i =(len(coords)-pos)/2
            output[0,0, x, y] = 0.5 ** i #decay
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

    def train(self, features, labels):
        output = self.forward(features)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
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
