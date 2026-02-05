#!/usr/bin/env python3
"""
MERTFORMER TITAN v1.0 [LIVE DEMO] — Snake (Autoplay)

Cyberpunk/Terminal aesthetic, self-playing AI, auto-restart on death.

Run:
  python snake_demo.py

Deps:
  pip install "pygame==2.6.1"
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import pygame


# ---------------------------
# Visual / Game Constants
# ---------------------------
CELL = 20
GRID_W = 32
GRID_H = 24
HEADER_H = 70

W = GRID_W * CELL
H = HEADER_H + GRID_H * CELL

FPS = 15

BG = (0, 0, 0)
GRID = (0, 22, 0)
NEON = (0, 255, 90)
NEON_DIM = (0, 140, 60)
FOOD = (255, 65, 65)
FOOD_DIM = (140, 20, 20)
TXT = (160, 255, 180)
TXT_DIM = (70, 160, 90)

HEADER = "MERTFORMER TITAN v1.0 [LIVE DEMO]"
REASONING = "Reasoning Speed: 30ms"
TOKENS = "Tokens: 1.58b"


DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
OPPOSITE = {(0, -1): (0, 1), (0, 1): (0, -1), (-1, 0): (1, 0), (1, 0): (-1, 0)}


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _in_bounds(p: tuple[int, int]) -> bool:
    return 0 <= p[0] < GRID_W and 0 <= p[1] < GRID_H


def _flood_fill_area(start: tuple[int, int], blocked: set[tuple[int, int]]) -> int:
    if start in blocked or not _in_bounds(start):
        return 0
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    area = 0
    while q:
        x, y = q.popleft()
        area += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n in seen or n in blocked or not _in_bounds(n):
                continue
            seen.add(n)
            q.append(n)
    return area


class SimulatedAI:
    """Simple heuristic AI: chase food, avoid collisions, prefer open space."""

    def choose(self, snake: list[tuple[int, int]], direction: tuple[int, int], food: tuple[int, int]) -> tuple[int, int]:
        head = snake[0]
        body = set(snake)

        candidates: list[tuple[int, int]] = []
        for d in DIRS.values():
            if len(snake) > 1 and d == OPPOSITE.get(direction):
                continue
            nx = head[0] + d[0]
            ny = head[1] + d[1]
            np = (nx, ny)
            if not _in_bounds(np):
                continue
            # Tail moves unless we eat. Treat tail as free in non-eat moves.
            tail = snake[-1]
            will_eat = np == food
            if np in body and not (not will_eat and np == tail):
                continue
            candidates.append(d)

        if not candidates:
            return direction

        best = candidates[0]
        best_score = -10**18

        for d in candidates:
            np = (head[0] + d[0], head[1] + d[1])
            will_eat = np == food
            # Simulate body after move
            if will_eat:
                new_snake = [np] + snake
            else:
                new_snake = [np] + snake[:-1]

            blocked = set(new_snake)
            area = _flood_fill_area(np, blocked)
            dist = _manhattan(np, food)

            # Heuristic:
            # - eating is top priority
            # - otherwise move closer to food
            # - prefer moves with more reachable free area (anti-trap)
            score = 0.0
            score += 10_000.0 if will_eat else 0.0
            score += -3.0 * float(dist)
            score += 0.25 * float(area)

            # Small bias: keep moving forward if equivalent.
            if d == direction:
                score += 0.5

            # Safety: avoid squeezing into areas smaller than the snake.
            if area < len(new_snake):
                score -= 500.0

            if score > best_score:
                best_score = score
                best = d

        return best


@dataclass
class GameState:
    snake: list[tuple[int, int]]
    direction: tuple[int, int]
    food: tuple[int, int]
    score: int


def _rand_food(occupied: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        p = (random.randrange(GRID_W), random.randrange(GRID_H))
        if p not in occupied:
            return p


def _new_game() -> GameState:
    cx = GRID_W // 2
    cy = GRID_H // 2
    snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
    direction = DIRS["RIGHT"]
    food = _rand_food(set(snake))
    return GameState(snake=snake, direction=direction, food=food, score=0)


def _cell_rect(p: tuple[int, int]) -> pygame.Rect:
    x, y = p
    return pygame.Rect(x * CELL, HEADER_H + y * CELL, CELL, CELL)


def _draw_grid(surface: pygame.Surface) -> None:
    # Subtle grid
    for x in range(GRID_W + 1):
        px = x * CELL
        pygame.draw.line(surface, GRID, (px, HEADER_H), (px, H), 1)
    for y in range(GRID_H + 1):
        py = HEADER_H + y * CELL
        pygame.draw.line(surface, GRID, (0, py), (W, py), 1)


def _draw_scanlines(surface: pygame.Surface) -> None:
    # Light scanline overlay (terminal vibe)
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    for y in range(0, H, 4):
        pygame.draw.line(overlay, (0, 0, 0, 40), (0, y), (W, y), 1)
    surface.blit(overlay, (0, 0))


def _load_font(size: int) -> pygame.font.Font:
    # Prefer "technical/retro" fonts; fallback to default monospace.
    for name in ("Cascadia Code", "Consolas", "Menlo", "Courier New"):
        f = pygame.font.SysFont(name, size)
        if f is not None:
            return f
    return pygame.font.SysFont(None, size)


def main() -> int:
    pygame.init()
    pygame.display.set_caption("MertFormer Titan — LIVE DEMO")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    font_h = _load_font(24)
    font_t = _load_font(18)

    ai = SimulatedAI()
    state = _new_game()

    last_restart = time.time()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return 0
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return 0

        # AI decides next move
        state.direction = ai.choose(state.snake, state.direction, state.food)

        head = state.snake[0]
        new_head = (head[0] + state.direction[0], head[1] + state.direction[1])

        dead = False
        if not _in_bounds(new_head):
            dead = True
        else:
            body = set(state.snake)
            tail = state.snake[-1]
            will_eat = new_head == state.food
            if new_head in body and not (not will_eat and new_head == tail):
                dead = True

        if dead:
            # Auto-restart immediately
            state = _new_game()
            last_restart = time.time()
        else:
            if new_head == state.food:
                state.snake = [new_head] + state.snake
                state.score += 1
                state.food = _rand_food(set(state.snake))
            else:
                state.snake = [new_head] + state.snake[:-1]

        # Render
        screen.fill(BG)

        # Header bar
        pygame.draw.rect(screen, (0, 10, 0), pygame.Rect(0, 0, W, HEADER_H))
        title = font_h.render(HEADER, True, TXT)
        screen.blit(title, (16, 10))

        # Telemetry
        score_txt = f"Score: {state.score}"
        t1 = font_t.render(REASONING, True, TXT_DIM)
        t2 = font_t.render(TOKENS, True, TXT_DIM)
        t3 = font_t.render(score_txt, True, TXT)
        screen.blit(t1, (16, 40))
        screen.blit(t2, (260, 40))
        screen.blit(t3, (W - t3.get_width() - 16, 40))

        _draw_grid(screen)

        # Food (glow)
        fr = _cell_rect(state.food)
        pygame.draw.rect(screen, FOOD_DIM, fr.inflate(10, 10), border_radius=6)
        pygame.draw.rect(screen, FOOD, fr.inflate(2, 2), border_radius=4)

        # Snake (glow-ish)
        for idx, seg in enumerate(state.snake):
            r = _cell_rect(seg)
            if idx == 0:
                pygame.draw.rect(screen, NEON_DIM, r.inflate(10, 10), border_radius=6)
                pygame.draw.rect(screen, NEON, r.inflate(2, 2), border_radius=4)
            else:
                pygame.draw.rect(screen, (0, 70, 30), r.inflate(6, 6), border_radius=6)
                pygame.draw.rect(screen, NEON_DIM, r.inflate(2, 2), border_radius=4)

        # Restart hint (subtle)
        if time.time() - last_restart < 1.0:
            hint = font_t.render("AUTO-RESTART", True, (80, 200, 120))
            screen.blit(hint, (W - hint.get_width() - 16, 10))

        _draw_scanlines(screen)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    raise SystemExit(main())

