from dataclasses import dataclass
from typing import override
import pygame
import sys
from abc import abstractmethod, ABC, ABCMeta
import json

import torch
from torch import nn
from torchvision.transforms import v2

import math

### TODO Batching
### TODO more than one training epoch
### TODO Shuffling
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

    # only run AI when new click happens
    if predictor.clicks >= numberCoords:
        return

    predictor.clicks = numberCoords

    need = predictor.inputLength + 2
    if numberCoords < need:
        return

    features_batch = []
    labels_batch = []

    # Slide backward in steps of 2 (xy pair)
    for end in range(numberCoords, 0, -2):
        start_features = end - predictor.inputLength - 2
        start_label    = end - 2

        if start_features < 0:
            break

        f = clicks[start_features:start_label]
        l = clicks[start_label:end]

        if len(f) != predictor.inputLength:
            continue
        if len(l) != 2:
            continue

        features_batch.append(f)
        labels_batch.append(l)

        if len(features_batch) >= 128:
            break

    if not features_batch:
        return

    features_batch.reverse()
    labels_batch.reverse()

    # ---- TRAIN ----
    predictor.fit_batch(features_batch, labels_batch)

    # ---- PREDICT (deterministic, dropout disabled) ----
    latest_features = features_batch[-1]
    prediction = predictor.predict([latest_features])

    print(f"prediction: {prediction}")

class SkipNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # stage 1
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout()
        )

        # stage 2
        self.block2 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout()
        )

        concat_dim = input_dim + 128 + 16

        self.head = nn.Sequential(
            nn.Linear(concat_dim, 2),
            # nn.Sigmoid()
        )

    def forward(self, x):
        h1 = self.block1(x)     # [B, 128]
        h2 = self.block2(h1)    # [B, 16]

        cat = torch.cat([x, h1, h2], dim=-1)

        return self.head(cat)




class NNPredictor(nn.Module):
    def __init__(self, game):
        super().__init__()

        self.game = game
        self.inputLength = 6
        self.clicks = 0

        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        print("device:", self.device)

        self.register_buffer(
            "normalization",
            torch.as_tensor(
                [game.window.width, game.window.height],
                dtype=torch.float32
            )
        )

        input_dim = self.inputLength

        self.model = SkipNet(input_dim)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        self.losses = []

        self.to(self.device)

    def loss(self, output, labels):
        return torch.mean(torch.abs(output - labels).pow(0.5))

    def forward(self, x):
        return self.model(x)

    def add_gradient_noise(self, std=1e-5):
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * std)

    # -------------------------------------------------
    # FIT BATCH (training mode)
    # -------------------------------------------------
    def fit_batch(
        self,
        features,
        labels,
        train_full=32,   # phase 1 steps
        train_end=16,     # phase 2 steps
        end_size=8       # refined subset size
    ):
        self.model.train()

        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels, dtype=torch.float32, device=self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        B = x.size(0)

        # -------------------------------
        # LR schedule across BOTH phases
        # -------------------------------
        total_steps = train_full + train_end
        lr_start = 1e-2
        lr_end   = 1e-3

        step_index = 0

        # ===============================
        # Phase 1 — FULL batch
        # ===============================
        for _ in range(train_full):
            t = step_index / (total_steps - 1)
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))

            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad()

            pred = self.forward(x)
            loss = self.loss(pred, y)

            self.losses.append(loss.detach())
            loss.backward()
            self.add_gradient_noise()
            self.optimizer.step()

            step_index += 1

        # ===============================
        # Phase 2 — SMALL refined subset
        # ===============================
        # take the most recent `end_size` samples (or all if smaller)
        k = min(end_size, B)
        xs = x[-k:]
        ys = y[-k:]

        for _ in range(train_end):
            t = step_index / (total_steps - 1)
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))

            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad()

            pred = self.forward(xs)
            loss = self.loss(pred, ys)

            self.losses.append(loss.detach())
            loss.backward()
            self.add_gradient_noise()
            self.optimizer.step()

            step_index += 1

        avg = sum(self.losses) / len(self.losses)

        print(
            f"loss: {loss.item():.4f}  "
            f"avg: {avg:.4f}  "
            f"lr_final: {lr:.6f}  "
            f"phase2_batch: {k}"
        )



    # -------------------------------------------------
    # PREDICT (dropout OFF)
    # -------------------------------------------------
    def predict(self, features):
        self.model.eval()     # <-- disables dropout

        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)

            if x.dim() == 1:
                x = x.unsqueeze(0)

            pred = self.forward(x)

        coords = pred[0].detach().cpu().numpy()

        self.game.ai = [int(coords[0]), int(coords[1])]
        self.game.ai_history.append(self.game.ai)

        return pred

        
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
            (255, 0, 255),
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
