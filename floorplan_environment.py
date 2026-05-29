import os
import random

import cv2
import numpy as np
import pygame
from PIL import Image
from scipy.ndimage import label


# Room colors.
LIVING = (255, 201, 71)
BED1 = (242, 123, 123)
BATH1 = (142, 202, 230)
BED2 = (255, 111, 97)
BATH2 = (255, 213, 194)

# Actions:
# 0: left
# 1: right
# 2: up
# 3: down
# 4: place

WIDTH = 50
HEIGHT = 50
SCALE_FACTOR = 1

INITIAL_POSITION = [12, 12]
ROOM_COLORS = [LIVING, BED1, BATH1, BED2, BATH2]
REQUIRED_CONNECTIONS = [
    (LIVING, BED1),
    (BED1, BATH1),
    (LIVING, BATH2),
    (LIVING, BED2),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLETED_DIR = os.path.join(BASE_DIR, "completed")
os.makedirs(COMPLETED_DIR, exist_ok=True)


class FloorplanPlacementEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH * SCALE_FACTOR, HEIGHT * SCALE_FACTOR))
        pygame.display.set_caption("Floorplan Layout Environment")
        self.clock = pygame.time.Clock()

        self.current_box = 0
        self.current_pos = INITIAL_POSITION.copy()
        self.placed_boxes = []
        self.steps = 0

        self.box_sizes, self.box_colors = self._create_random_room_sequence()

    def _create_random_room_sequence(self):
        rooms = [
            (random.randint(15, 20), random.randint(15, 20)),  # living room
            (random.randint(10, 12), random.randint(10, 12)),  # bedroom 1
            (random.randint(4, 6), random.randint(4, 6)),      # bathroom 1
            (random.randint(8, 12), random.randint(8, 12)),    # bedroom 2
            (random.randint(4, 6), random.randint(4, 6)),      # bathroom 2
        ]
        room_pairs = list(zip(rooms, ROOM_COLORS))
        random.shuffle(room_pairs)
        sizes, colors = zip(*room_pairs)
        return list(sizes), list(colors)

    def reset(self):
        self.screen.fill((0, 0, 0))
        self.current_box = 0
        self.current_pos = INITIAL_POSITION.copy()
        self.placed_boxes = []
        self.steps = 0
        self.box_sizes, self.box_colors = self._create_random_room_sequence()
        return self._state_array()

    def step(self, action):
        if self.steps == 0:
            action = 4
        self.steps += 1

        reward = 0
        done = False

        new_pos = self._next_position(action)
        if new_pos is not None and self._can_move_to(new_pos):
            self.current_pos = new_pos

        if action == 4:
            return self._place_current_box()

        self._draw()
        return self._state_with_current_box(), reward, done, {}

    def _next_position(self, action):
        if action == 0:
            return [self.current_pos[0] - 1, self.current_pos[1]]
        if action == 1:
            return [self.current_pos[0] + 1, self.current_pos[1]]
        if action == 2:
            return [self.current_pos[0], self.current_pos[1] - 1]
        if action == 3:
            return [self.current_pos[0], self.current_pos[1] + 1]
        return None

    def _can_move_to(self, new_pos):
        if self.current_box >= len(self.box_sizes):
            return False

        current_size = self.box_sizes[self.current_box]
        inside_bounds = (
            0 <= new_pos[0] < WIDTH - current_size[0] + 1
            and 0 <= new_pos[1] < HEIGHT - current_size[1] + 1
        )
        return inside_bounds and self._is_adjacent_to_existing_box(new_pos)

    def _place_current_box(self):
        if self.current_box >= len(self.box_sizes):
            return self._state_array(), 0, True, {"layout_complete": True}

        if self._is_current_box_out_of_bounds():
            print("out of bound")
            return self._state_with_current_box(), -10, True, {
                "layout_complete": False,
                "termination_reason": "out_of_bounds",
            }

        self.placed_boxes.append((self.current_box, tuple(self.current_pos)))
        self.current_box += 1
        self._draw()

        state = self._state_with_current_box()
        reward = self._layout_reward(self._state_array())
        done = self.current_box == len(self.box_sizes)
        return state, reward, done, {
            "layout_complete": done,
            "placed_rooms": self.current_box,
        }

    def _is_current_box_out_of_bounds(self):
        size = self.box_sizes[self.current_box]
        return (
            self.current_pos[0] + size[0] > WIDTH
            or self.current_pos[1] + size[1] > HEIGHT
        )

    def _layout_reward(self, state_rgb):
        reward = self._connection_reward(state_rgb)
        reward -= self._overlap_penalty()
        return reward

    def _connection_reward(self, state_rgb):
        placed_colors = {
            color
            for color in ROOM_COLORS
            if np.any(np.all(state_rgb == np.array(color), axis=-1))
        }

        reward = 0
        for color1, color2 in REQUIRED_CONNECTIONS:
            if color1 in placed_colors and color2 in placed_colors:
                reward += self._connection_score(state_rgb, color1, color2)

        if reward == len(REQUIRED_CONNECTIONS):
            print("perfect")
            filename = f"perfect_{random.randint(1, 1000000)}.png"
            Image.fromarray(state_rgb).save(os.path.join(COMPLETED_DIR, filename))

        return reward

    def _connection_score(self, image, color1, color2):
        component_count = self._connected_component_count(image, color1, color2)
        if component_count == 1:
            return 1
        if component_count > 1:
            return 1 / component_count
        return 0

    def _connected_component_count(self, image, color1, color2):
        mask1 = np.all(image == np.array(color1), axis=-1)
        mask2 = np.all(image == np.array(color2), axis=-1)

        if not mask1.any() or not mask2.any():
            return 0

        combined_mask = np.logical_or(mask1, mask2)
        _, num_features = label(combined_mask)
        return num_features

    def _overlap_penalty(self):
        penalty = 0
        for i in range(len(self.placed_boxes)):
            box1 = self._box_rect(self.placed_boxes[i])
            for j in range(i + 1, len(self.placed_boxes)):
                box2 = self._box_rect(self.placed_boxes[j])
                penalty += self._calculate_iou(box1, box2) * 4
        return penalty

    def _box_rect(self, placed_box):
        box_index, position = placed_box
        return position + self.box_sizes[box_index]

    def _is_adjacent_to_existing_box(self, new_pos):
        if self.current_box == 0:
            return True

        current_size = self.box_sizes[self.current_box]
        for box, pos in self.placed_boxes:
            box_size = self.box_sizes[box]
            overlaps_or_touches = (
                new_pos[0] + current_size[0] >= pos[0]
                and new_pos[0] <= pos[0] + box_size[0]
                and new_pos[1] + current_size[1] >= pos[1]
                and new_pos[1] <= pos[1] + box_size[1]
            )
            if overlaps_or_touches:
                return True
        return False

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x_inter_left = max(x1, x2)
        y_inter_top = max(y1, y2)
        x_inter_right = min(x1 + w1, x2 + w2)
        y_inter_bottom = min(y1 + h1, y2 + h2)

        if x_inter_right < x_inter_left or y_inter_bottom < y_inter_top:
            return 0.0

        inter_area = (x_inter_right - x_inter_left) * (y_inter_bottom - y_inter_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        return inter_area / float(box1_area + box2_area - inter_area)

    def _state_array(self):
        state_rgb = np.array(pygame.surfarray.array3d(self.screen))
        return np.transpose(state_rgb, (1, 0, 2))

    def _state_with_current_box(self):
        state = self._state_array()
        if self.current_box >= len(self.box_sizes):
            return state

        state_with_box = state.copy()
        x, y = self.current_pos
        w, h = self.box_sizes[self.current_box]
        cv2.rectangle(state_with_box, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
        return state_with_box

    def _draw(self):
        self.screen.fill((0, 0, 0))
        for index, (box, pos) in enumerate(self.placed_boxes):
            scaled_pos = [coord * SCALE_FACTOR for coord in pos]
            scaled_size = [size * SCALE_FACTOR for size in self.box_sizes[box]]
            pygame.draw.rect(self.screen, self.box_colors[index], (*scaled_pos, *scaled_size))
        pygame.display.flip()
