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
import numpy as np

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


# class SkipNet(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()

#         # stage 1
#         self.block1 = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout()
#         )

#         # stage 2
#         self.block2 = nn.Sequential(
#             nn.Linear(128, 16),
#             nn.ReLU(),
#             nn.Dropout()
#         )

#         concat_dim = input_dim + 128 + 16

#         self.head = nn.Sequential(
#             nn.Linear(concat_dim, 2),
#             # nn.Sigmoid()
#         )

#     def forward(self, x):
#         h1 = self.block1(x)     # [B, 128]
#         h2 = self.block2(h1)    # [B, 16]

#         cat = torch.cat([x, h1, h2], dim=-1)

#         return self.head(cat)


# class SkipNet(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         lstm_hidden: int = 16,   # tiny!
#         proj_dim: int = 128,     # richer features after concat
#         p_drop: float = 0.2,
#     ):
#         super().__init__()

#         assert input_dim % 2 == 0, \
#             "input_dim must be even: packed as [x, y, x, y, ...]"

#         self.input_dim = input_dim
#         self.step_dim = 2
#         self.timesteps = input_dim // 2

#         # ---- tiny LSTM over (x,y) pairs ----
#         self.lstm = nn.LSTM(
#             input_size=self.step_dim,
#             hidden_size=lstm_hidden,
#             num_layers=1,
#             batch_first=True,
#         )
#         self.drop1 = nn.Dropout(p=p_drop)

#         # ---- project ALL LSTM outputs ----
#         # LSTM outputs: [B, T, lstm_hidden]
#         # Flatten to:  [B, T * lstm_hidden]
#         self.proj = nn.Sequential(
#             nn.Linear(self.timesteps * lstm_hidden, proj_dim),
#             nn.ReLU(),
#             nn.Dropout(p=p_drop),
#         )

#         # ---- Stage 2 ----
#         self.block2 = nn.Sequential(
#             nn.Linear(proj_dim, 16),
#             nn.ReLU(),
#             nn.Dropout(p=p_drop),
#         )

#         # ---- Head ----
#         concat_dim = input_dim + proj_dim + 16
#         self.head = nn.Linear(concat_dim, 2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, input_dim]
#            packed as [x0, y0, x1, y1, ..., x_{T-1}, y_{T-1}]
#         """
#         B, D = x.shape
#         assert D == self.input_dim

#         # ---- reshape to sequence ----
#         seq = x.view(B, self.timesteps, self.step_dim)   # [B, T, 2]

#         # ---- LSTM ----
#         out, _ = self.lstm(seq)                          # [B, T, 16]
#         out = self.drop1(out)

#         # flatten all timestep outputs
#         flat = out.reshape(B, -1)                        # [B, T*16]

#         # ---- projection ----
#         h1 = self.proj(flat)                             # [B, proj_dim]

#         # ---- stage 2 ----
#         h2 = self.block2(h1)                             # [B, 16]

#         # ---- skip connection ----
#         cat = torch.cat([x, h1, h2], dim=-1)

#         return self.head(cat)

class SkipNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        proj_dim: int = 128,
        p_drop: float = 0.5,
    ):
        super().__init__()

        assert input_dim % 2 == 0, \
            "input_dim must be even: packed as [x, y, x, y, ...]"

        self.input_dim = input_dim
        self.step_dim = 2
        self.timesteps = input_dim // 2

        # ===== cache sin/cos for exactly our length =====
        t = torch.arange(self.timesteps).float()
        sin = torch.sin(t * math.pi / self.timesteps)
        cos = torch.cos(t * math.pi / self.timesteps)

        self.register_buffer("sin_cache", sin.unsqueeze(-1))  # [T,1]
        self.register_buffer("cos_cache", cos.unsqueeze(-1))  # [T,1]

        # ===== tiny conv stack =====
        self.conv = nn.Sequential(
            nn.Conv1d(4, conv_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
        )

        # ===== projection after flatten conv output =====
        self.proj = nn.Sequential(
            nn.Linear(self.timesteps * conv_channels, proj_dim),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
        )

        # ===== head uses original input + conv projection =====
        concat_dim = input_dim + proj_dim
        self.head = nn.Linear(concat_dim, 2)

    def forward(self, x):
        B, D = x.shape
        assert D == self.input_dim

        # ---- reshape to [B,T,2] ----
        seq = x.view(B, self.timesteps, self.step_dim)

        # ---- add cached sin/cos ----
        sin = self.sin_cache.unsqueeze(0).expand(B, -1, -1)   # [B,T,1]
        cos = self.cos_cache.unsqueeze(0).expand(B, -1, -1)   # [B,T,1]
        seq = torch.cat([seq, sin, cos], dim=-1)              # [B,T,4]

        # ---- conv expects [B,C,T] ----
        seq = seq.permute(0, 2, 1)                            # [B,4,T]

        feat = self.conv(seq)                                 # [B,C,T]
        flat = feat.reshape(B, -1)                            # [B,T*C]

        h = self.proj(flat)                                   # [B,proj_dim]

        # ---- skip concat (original + conv features) ----
        cat = torch.cat([x, h], dim=-1)

        return self.head(cat)


class NNPredictor(nn.Module):
    def __init__(self, game):
        super().__init__()

        self.game = game
        self.inputLength = 8
        self.clicks = 0

        # single source of truth
        self.train_data_size = 64

        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        print("device:", self.device)

        # self.register_buffer(
        #     "normalization",
        #     torch.as_tensor(
        #         [game.window.width, game.window.height],
        #         dtype=torch.float32
        #     )
        # )

        input_dim = self.inputLength

        self.model = SkipNet(input_dim)
        # self.model.compile(dynamic=False)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        self.losses = []

        # ---------------------------------
        # persistent numpy rolling buffers
        # ---------------------------------
        self.max_samples = self.train_data_size
        self.sample_count = 0

        self.features_np = np.zeros(
            (self.max_samples, self.inputLength),
            dtype=np.float32
        )
        self.labels_np = np.zeros(
            (self.max_samples, 2),
            dtype=np.float32
        )

        self.to(self.device)

    # def loss(self, output, labels):
    #     return torch.mean(torch.abs(output - labels))

    def forward(self, x):
        return self.model(x)

    def add_gradient_noise(self, std=1e-5):
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is None:
                    continue

                g = p.grad

                # If NaN or Inf → kill it
                if not torch.isfinite(g).all():
                    print("NaN/Inf gradient detected → reset to 0")
                    p.grad.zero_()
                    raise ValueError("OVERFLOW")
                    continue

                noise = (torch.rand_like(g) * 2.0 - 1.0) * std  # uniform in [-std, std]
                g.add_(noise)

    # ======================================================
    # BUFFER MGMT
    # ======================================================
    def append_sample(self, features, label):
        if self.sample_count >= self.max_samples:
            # simple FIFO rollover
            self.features_np[:-1] = self.features_np[1:]
            self.labels_np[:-1]   = self.labels_np[1:]
            self.sample_count -= 1

        self.features_np[self.sample_count] = np.asarray(features, dtype=np.float32)
        self.labels_np[self.sample_count]   = np.asarray(label, dtype=np.float32)
        self.sample_count += 1

    # ======================================================
    # FIT RECENT
    # ======================================================
    def fit_recent(self):
        if self.sample_count == 0:
            return

        k = min(self.train_data_size, self.sample_count)

        x = torch.tensor(
            self.features_np[self.sample_count - k:self.sample_count],
            dtype=torch.float32,
            device=self.device
        )
        y = torch.tensor(
            self.labels_np[self.sample_count - k:self.sample_count],
            dtype=torch.float32,
            device=self.device
        )

        self.model.train()

        lr_start = 1e-2
        lr_end   = 1e-3
        train_full = 32
        train_end  = 16
        total_steps = train_full + train_end
        step_index = 0

        def weighted_loss(pred, target, bias):
            N = pred.size(0)
            idx = torch.arange(N, dtype=torch.float64, device=self.device)

            # shift so last = 1.0
            w = torch.exp(bias * (idx - (N - 1)))

            # normalize
            w = w / w.sum()

            res = torch.abs(pred - target)
            safe = res.clamp_min(1e-6)
            res = torch.minimum(safe.pow(0.8),res)

            per = torch.mean(res, dim=1)
            return torch.sum(per * w)


        # ---- phase 1 ----
        for _ in range(train_full):
            t = step_index / (total_steps - 1)
            lr = lr_end + 0.5*(lr_start - lr_end)*(1+math.cos(math.pi*t))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = weighted_loss(pred, y, 0.15)
            self.losses.append(loss.detach())
            loss.backward()
            self.add_gradient_noise()
            self.optimizer.step()
            step_index += 1

        # ---- phase 2 ----
        k2 = min(8, k)
        xs = x[-k2:]
        ys = y[-k2:]

        for _ in range(train_end):
            t = step_index / (total_steps - 1)
            lr = lr_end + 0.5*(lr_start - lr_end)*(1+math.cos(math.pi*t))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad()
            pred = self.model(xs)
            loss = weighted_loss(pred, ys, 0.3)
            self.losses.append(loss.detach())
            loss.backward()
            self.add_gradient_noise()
            self.optimizer.step()
            step_index += 1

        avg = sum(self.losses) / len(self.losses)

        print(
            f"train loss: {loss.item():.4f}  "
            # f"avg: {avg:.4f}  "
            # f"lr_final: {lr:.6f}  "
            # f"train_used: {k}"
        )

    # ======================================================
    # unified AI pipeline
    # ======================================================
    def update_predictor(self, marker):
        clicks = marker.ai_point_list
        numberCoords = len(clicks)

        if numberCoords <= self.clicks:
            return

        self.clicks = numberCoords

        need = self.inputLength + 2
        if numberCoords < need:
            return

        # build ONLY new training windows
        for end in range(numberCoords, 0, -2):
            start_f = end - self.inputLength - 2
            start_l = end - 2
            if start_f < 0:
                break

            f = clicks[start_f:start_l]
            l = clicks[start_l:end]

            if len(f) == self.inputLength and len(l) == 2:
                self.append_sample(f, l)

        self.fit_recent()

        # ---- predict newest ----
        latest = clicks[-self.inputLength:]
        x = torch.tensor([latest], dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)

        coords = pred[0].detach().cpu().numpy()
        self.game.ai = [int(coords[0]), int(coords[1])]
        self.game.ai_history.append(self.game.ai)

        print("prediction:", pred)



# ------------------------------------------------------
# compatibility wrapper for old callsite
# ------------------------------------------------------
def predictions(predictor, marker, game):
    predictor.update_predictor(marker)


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
