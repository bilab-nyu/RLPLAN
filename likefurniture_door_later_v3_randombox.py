# Can I use multi-agent that have different kinds of actions?

# Define box sizes action is needed
import subprocess, shlex
import os
from osgeo import gdal, ogr, osr
import copy
import cv2
import pygame
import sys
import numpy as np
from PIL import Image
from datetime import datetime
import random
from scipy.ndimage import binary_dilation

from osgeo.gdalconst import GA_ReadOnly
import matplotlib.pyplot as plt

pygame.init()

# KITCHEN = (65, 166, 156)

# Define colors
LIVING = (255, 201, 71)
DOOR = (100,100,100)
BED1 = (242, 123, 123)
BATH1 = (142, 202, 230)
BED2 = (255, 111, 97)
BATH2 = (255, 213, 194)

DOOR = (100,100,100)
DOOR_SIZE = (1,1)

# Define box sizes
# action
# 0: left
# 1: right
# 2: up
# 3: down
# 4: place

# Define game environment dimensions
WIDTH = 50
HEIGHT = 50

scale_factor = 1

import os
import shlex
import subprocess
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
from osgeo import gdal, ogr, osr
from scipy.ndimage import binary_dilation, label

def export_layout_to_dxf(screen_array: np.ndarray,
                         output_dir: str,
                         door_colors: list[tuple],
                         wall_color: tuple = (255, 0, 0),
                         temp_prefix: str = "layout"):
    """
    Given an RGB numpy array of the final layout (screen_array),
    runs through the pipeline: raster → contours → polygonize → DXF.
    Returns path to the generated DXF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1. Save raw state
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    raw_png = os.path.join(output_dir, f"{temp_prefix}_{ts}.png")
    Image.fromarray(screen_array).save(raw_png)

    # 2. Binary threshold
    gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bw_path = os.path.join(output_dir, f"{temp_prefix}_bw_{ts}.png")
    cv2.imwrite(bw_path, bw)

    # 3. Extract external wall contour
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ext_wall = np.zeros_like(bw)
    cv2.drawContours(ext_wall, contours, -1, 255, 1)
    ext_path = os.path.join(output_dir, f"{temp_prefix}_ext_{ts}.png")
    cv2.imwrite(ext_path, ext_wall)

    # 4. Build door mask
    door_mask = np.zeros_like(bw, dtype=bool)
    for dc in door_colors:
        m = np.all(screen_array == np.array(dc), axis=-1)
        door_mask |= m
    door_mask = binary_dilation(door_mask, structure=np.ones((3,3),bool))
    door_png = os.path.join(output_dir, f"{temp_prefix}_door_{ts}.png")
    Image.fromarray((door_mask * 255).astype(np.uint8)).save(door_png)

    # 5. Combine walls + doors into a single mask for polygonizing
    combined = ((ext_wall > 0) | door_mask).astype(np.uint8) * 255
    comb_path = os.path.join(output_dir, f"{temp_prefix}_mask_{ts}.png")
    Image.fromarray(combined).save(comb_path)

    # 6. Polygonize via GDAL → Shapefile
    shp_path = os.path.join(output_dir, f"{temp_prefix}.shp")
    ds = gdal.Open(comb_path, gdal.GA_ReadOnly)
    # assume identity geoTransform if none
    gt = ds.GetGeoTransform() or (0,1,0,0,0,-1)
    ds.SetGeoTransform(gt)
    srcband = ds.GetRasterBand(1)
    # create shapefile
    drv = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = drv.CreateDataSource(shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjectionRef())
    lyr = shp_ds.CreateLayer("poly", srs=srs)
    fld = ogr.FieldDefn("ID", ogr.OFTInteger)
    lyr.CreateField(fld)
    gdal.Polygonize(srcband, None, lyr, 0, [], callback=None)
    shp_ds.Destroy()
    ds = None

    # 7. Call external converters
    subprocess.run(shlex.split("make_dxf.bat"), cwd=output_dir)
    subprocess.run(shlex.split("make_vga.bat"), cwd=output_dir)

    dxf_path = os.path.join(output_dir, "layout.dxf")
    return dxf_path


def draw_rect_centered(surface, color, center, size):
    x = center[0] - size[0] // 2
    y = center[1] - size[1] // 2
    pygame.draw.rect(surface, color, (x, y, *size))

class layoutAI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH * scale_factor, HEIGHT * scale_factor))
        pygame.display.set_caption("Box Placement Game")
        self.clock = pygame.time.Clock()
        self.is_placing_door = False  # Indicates if the agent is currently placing a door

        self.current_box = 0
        self.current_pos = [12,12]

        self.placed_boxes = []
        self.door_tf = False
        self.steps = 0
        self.case_types = []
        self.flag = 0
        self.no_door_image = None
        self.prev_conn = 0
        rooms = [
            (random.randint(15,20), random.randint(15,20)),  # livingroom
            (random.randint(10,12), random.randint(10,12)),  # bed1
            (random.randint(4,6),  random.randint(4,6)),       # bath1
            (random.randint(8,12), random.randint(8,12)),      # bed2
            (random.randint(4,6),  random.randint(4,6))        # bath2
        ]
        rooms_colors = [LIVING, BED1, BATH1, BED2, BATH2]

        doors = [DOOR_SIZE, DOOR_SIZE, DOOR_SIZE, DOOR_SIZE]
        doors_colors = [DOOR, DOOR, DOOR, DOOR]

        # 방의 크기와 해당 색상을 함께 묶은 후 셔플합니다.
        room_pairs = list(zip(rooms, rooms_colors))
        random.shuffle(room_pairs)
        rooms, rooms_colors = zip(*room_pairs)  # 셔플된 결과를 분리

        # 셔플된 방과 원래 순서의 문을 합칩니다.
        self.BOX_SIZES = list(rooms) + doors
        self.BOX_COLORS = list(rooms_colors) + doors_colors
        
    def normalize_reward(self, reward):
        # 만약 처음 보상 업데이트라면 초기화
        if not hasattr(self, 'reward_count'):
            self.reward_count = 0
            self.reward_sum = 0.0
            self.reward_square_sum = 0.0
        # 보상 누적값 업데이트
        self.reward_count += 1
        self.reward_sum += reward
        self.reward_square_sum += reward ** 2
        # 평균 계산
        mean_reward = self.reward_sum / self.reward_count
        # 분산과 표준편차 계산 (분산이 0이면 std=1로 처리)
        variance = self.reward_square_sum / self.reward_count - mean_reward ** 2
        std = np.sqrt(variance) if variance > 0 else 1.0
        # 정규화: (reward - mean) / (std + 작은값)
        normalized_reward = (reward - mean_reward) / (std + 1e-8)
        return normalized_reward

    def draw_contour(self,path):
        im_arr = cv2.imread(path)
        LIVING = [71, 201, 255]
        BED1 = [123, 123, 242]
        # BATH1 = [230, 202, 142]
        BED2 = [97, 111, 255]
        # BATH2 = [194, 213, 255]

        # unique_colors = np.array([LIVING,BED1,BATH1,BED2,BATH2]).astype(np.uint8)
        unique_colors = np.array([LIVING,BED1,BED2]).astype(np.uint8)

        # Create a copy of the original image to draw the white contours
        contour_image = im_arr.copy()

        # Kernel for erosion
        kernel = np.ones((3,3), np.uint8)

        for color in unique_colors:
            mask = np.all(im_arr == color, axis=-1)
            mask_uint8 = (mask * 255).astype(np.uint8)
            eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
            contour_edges = mask_uint8 - eroded_mask
            contour_image[contour_edges == 255] = [255, 255, 255]
        return contour_image
    
    def draw_contour2(self,path):
        image = cv2.imread(path)
        unique_colors = np.array([0,0,0]).astype(np.int8)

        # Create a copy of the original image to draw the white contours
        contour_image = image.copy()

        # Kernel for erosion
        kernel = np.ones((3,3), np.uint8)

        for color in unique_colors:
            mask = np.all(image == color, axis=-1)
            mask_uint8 = (mask * 255).astype(np.uint8)
            eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
            contour_edges = mask_uint8 - eroded_mask
            contour_image[contour_edges == 255] = [0, 0, 255]
        return contour_image


    def door_random_allocation(case_type,x0,y0,w0,h0,x1,y1,w1,h1):
        if case_type == "type1":
            x, y = (random.randint(x1 + 1, x0 + w0 - 1), y1) if random.choice(['horizontal', 'vertical']) == 'horizontal' else (x1, random.randint(y1 + 1, y0 + h0 - 1))
            return (x,y)
        if case_type == "type2":
            x, y = (random.randint(x0 + 1, x1 + w1 - 1), y1) if random.choice(['first', 'second']) == 'first' else (x1 + w1, random.randint(y1 + 1, y0 + h0 - 1))
            return (x,y)
        if case_type == "type3":
            x, y = (random.randint(x1 + 1, x0 + w0 - 1), y1 + h1) if random.choice(['horizontal', 'vertical']) == 'horizontal' else (x1, random.randint(y0 + 1, y1 + h1 - 1))
            return (x,y)
        if case_type == "type4":
            x, y = (random.randint(x0 + 1, x1 + w1 - 1), y1 + h1) if random.choice(['horizontal', 'vertical']) == 'horizontal' else (x1 + w1, random.randint(y0 + 1, y1 + h1 - 1))
            return (x,y)
        
    def check_adjacent(self, image, color1, color2):
        from scipy.ndimage import label, binary_dilation
        mask1 = np.all(image == np.array(color1), axis=-1)
        mask2 = np.all(image == np.array(color2), axis=-1)

        if not mask1.any() or not mask2.any():
            return 0, None  # 0은 실패로 간주

        combined_mask = np.logical_or(mask1, mask2)

        labeled_array, num_features = label(combined_mask)
        return num_features
    
# coordinate-based box adjacency check
    def is_box_adjacent(self,newpos):
        BOX_SIZES = self.BOX_SIZES
        if self.current_box == 0:
            return True
        current_size = BOX_SIZES[self.current_box]
        for box, pos in self.placed_boxes:
            box_size = BOX_SIZES[box]
            if (newpos[0] + current_size[0] >= pos[0] and newpos[0] <= pos[0] + box_size[0] and
                newpos[1] + current_size[1] >= pos[1] and newpos[1] <= pos[1] + box_size[1]):
                return True
        return False

    def contains_target_colors(self, window, color1, color2):
        # Check if both colors are present in the window
        has_color1 = np.any(np.all(window == color1, axis=-1))
        has_color2 = np.any(np.all(window == color2, axis=-1))
        return has_color1 and has_color2
    
    def matching_windows(self,state,color1,color2):
        matching_windows = []
        height, width, _ = state.shape
        for y in range(height - 1):
            for x in range(width - 1):
                window = state[y:y+2, x:x+2]
                # Check if the window contains both target colors
                if self.contains_target_colors(window, color1, color2):
                    matching_windows.append((y, x))
        return matching_windows

    def calculate_iou(self, box1, box2):
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

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def reset(self):
        self.screen.fill((0, 0, 0))
        self.current_box = 0
        self.current_pos = [12, 12]
        self.placed_boxes = []
        self.steps = 0
        self.flag = 0
        self.prev_conn = 0
        # 방(rooms)과 색상(rooms_colors) 정의
        rooms = [
            (random.randint(15,20), random.randint(15,20)),  # livingroom
            (random.randint(10,12), random.randint(10,12)),  # bed1
            (random.randint(4,6),  random.randint(4,6)),       # bath1
            (random.randint(8,12), random.randint(8,12)),      # bed2
            (random.randint(4,6),  random.randint(4,6))         # bath2
        ]
        rooms_colors = [LIVING, BED1, BATH1, BED2, BATH2]

        # 방과 색상을 묶어서 셔플
        room_pairs = list(zip(rooms, rooms_colors))
        random.shuffle(room_pairs)
        rooms, rooms_colors = zip(*room_pairs)

        # 문의 경우는 그대로 유지
        doors = [DOOR_SIZE, DOOR_SIZE, DOOR_SIZE, DOOR_SIZE]
        doors_colors = [DOOR, DOOR, DOOR, DOOR]

        # 셔플된 방과 원래 문의 순서를 합침
        self.BOX_SIZES = list(rooms) + doors
        self.BOX_COLORS = list(rooms_colors) + doors_colors
        return np.array(pygame.surfarray.array3d(self.screen))

    def step(self, action):
        BOX_SIZES = self.BOX_SIZES
        if self.steps == 0:
            action = 4
        self.steps += 1
        done = False
        reward = 0
        if action == 0:
            new_pos = [self.current_pos[0] - 1, self.current_pos[1]]
        elif action == 1:
            new_pos = [self.current_pos[0] + 1, self.current_pos[1]]
        elif action == 2:
            new_pos = [self.current_pos[0], self.current_pos[1] - 1]
        elif action == 3:
            new_pos = [self.current_pos[0], self.current_pos[1] + 1]
        else:
            new_pos = None
            
        # print(f"Action: {action}, Current Pos: {self.current_pos}, New Pos: {new_pos}") #added print statement

        # can move inside of the rooms before door allocation
        if len(self.placed_boxes) < 5:
            if new_pos is not None and self.is_box_adjacent(new_pos):
                if 0 <= new_pos[0] < WIDTH - BOX_SIZES[self.current_box][0] + 1 and 0 <= new_pos[1] < HEIGHT - BOX_SIZES[self.current_box][1] + 1:
                    self.current_pos = new_pos
                    
        # room 이 placed 된다.
        if action == 4:
            if self.current_pos[0] + BOX_SIZES[self.current_box][0] > WIDTH or self.current_pos[1] + BOX_SIZES[self.current_box][1] > WIDTH:
                print("out of bound")
                reward = -10
                state_rgb = np.array(pygame.surfarray.array3d(self.screen))
                state_rgb = np.transpose(state_rgb, (1, 0, 2))  # (height, width, 3)로 맞춤
                
                state_rgb_with_box = state_rgb.copy()

                y, x = self.current_pos[1], self.current_pos[0]  # (x, y) 좌표 조정
                h, w = BOX_SIZES[self.current_box][1], BOX_SIZES[self.current_box][0]
                
                cv2.rectangle(state_rgb_with_box, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
                state = state_rgb_with_box
                done = True
                return state, reward, done, {}
            self.placed_boxes.append((self.current_box, tuple(self.current_pos)))
            self.current_box += 1
            self._draw()
            
            state_rgb = np.array(pygame.surfarray.array3d(self.screen))
            state_rgb = np.transpose(state_rgb, (1, 0, 2))  # (height, width, 3)로 맞춤
            
            state_rgb_with_box = state_rgb.copy()

            y, x = self.current_pos[1], self.current_pos[0]  # (x, y) 좌표 조정
            h, w = BOX_SIZES[self.current_box][1], BOX_SIZES[self.current_box][0]
            
            cv2.rectangle(state_rgb_with_box, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
            state = state_rgb_with_box
            
            if self.current_box < 6:
                
                living_bed1 = 0
                bed1_bath1 = 0
                living_bath2 = 0
                living_bed2 = 0
                
                # placed_boxes에는 (box_index, position) 튜플들이 들어있음.
                # room들은 box_index가 total_rooms 미만인 것으로 가정
                # rendered state 이미지(state)에서 각 room의 색상이 존재하는지 검사
                placed_rooms_info = {}
                for room_color in [LIVING, BED1, BATH1, BED2, BATH2]:
                    # state가 (height, width, 3) 형태라고 가정합니다.
                    mask = np.all(state_rgb == np.array(room_color), axis=-1)
                    if np.any(mask):
                        placed_rooms_info[room_color] = True
                # 예를 들어, LIVING과 BED1의 연결성을 확인하고 싶다면
                # mask1 = np.all(state_rgb == np.array(LIVING), axis=-1)
                # mask2 = np.all(state_rgb == np.array(BED1), axis=-1)
                # print(f"mask1: {mask1.any()}; mask2: {mask2.any()}")
                # print(placed_rooms_info)
                if LIVING in placed_rooms_info and BED1 in placed_rooms_info:
                    # Image.fromarray(state_rgb).save("state_test_living_bed1.png")
                    living_bed1 = self.check_adjacent(np.array(state_rgb), LIVING, BED1)
                    if living_bed1 == 1:
                        living_bed1 = 1
                    elif living_bed1 > 1:
                        living_bed1 = 1 / living_bed1
                # 다른 pair들도 동일한 방식으로 검사할 수 있습니다.
                if BED1 in placed_rooms_info and BATH1 in placed_rooms_info:
                    # Image.fromarray(state_rgb).save("state_test_bed1_bath1.png")
                    bed1_bath1 = self.check_adjacent(np.array(state_rgb), BED1, BATH1)
                    if bed1_bath1 == 1:
                        bed1_bath1 = 1
                    elif bed1_bath1 > 1:
                        bed1_bath1 = 1 / bed1_bath1
                    
                if LIVING in placed_rooms_info and BATH2 in placed_rooms_info:
                    # Image.fromarray(state_rgb).save("state_test_liv_bath2.png")
                    living_bath2 = self.check_adjacent(np.array(state_rgb), LIVING, BATH2)
                    if living_bath2 == 1:
                        living_bath2 = 1
                    elif living_bath2 > 1:
                        living_bath2 = 1 / living_bath2

                if LIVING in placed_rooms_info and BED2 in placed_rooms_info:
                    # Image.fromarray(state_rgb).save("state_test_liv_bed2.png")
                    living_bed2 = self.check_adjacent(np.array(state_rgb), LIVING, BED2)
                    if living_bed2 == 1:
                        living_bed2 = 1
                    elif living_bed2 > 1:
                        living_bed2 = 1 / living_bed2

                conn = living_bed1 + bed1_bath1 + living_bath2 + living_bed2
                # print(f"conn score: {conn}")
                if conn == 4:
                    print("perfect")
                    Image.fromarray(state_rgb).save(f"completed/perfect_{random.randint(1,1000000)}.png")

                # reward add (adj)
                reward = conn
                
                # giving reward by iou
                for i in range(len(self.placed_boxes)):
                    box1 = self.placed_boxes[i][1] + BOX_SIZES[self.placed_boxes[i][0]]
                    for j in range(i + 1, len(self.placed_boxes)):
                        box2 = self.placed_boxes[j][1] + BOX_SIZES[self.placed_boxes[j][0]]
                        iou = self.calculate_iou(box1, box2)
                        reward -= iou*4
                # print(reward)
                # reward = self.normalize_reward(reward)
                if self.current_box == 5:
                    done = True
                return state, reward, done, {}
            
            # when the placement ends, decide the position of the next room
            if len(self.placed_boxes) < 5:
                box_now = self.current_box - 1
                box_now_size = BOX_SIZES[box_now]
                box_next_size = BOX_SIZES[self.current_box]
                self.current_pos = (self.current_pos[0] + box_now_size[0]//2 - box_next_size[0]//2,  self.current_pos[1] + box_now_size[1]//2 - box_next_size[1]//2)
                # print(f"{box_now} and {self.current_box} are placed")
                # print(f"New position after placement: {self.current_pos}") #added print statement
            else:
                done = True
                # door placement 포함.
                pass
        self._draw()
        state_rgb = np.array(pygame.surfarray.array3d(self.screen))
        state_rgb = np.transpose(state_rgb, (1, 0, 2))  # (height, width, 3)로 맞춤
        
        state_rgb_with_box = state_rgb.copy()

        y, x = self.current_pos[1], self.current_pos[0]  # (x, y) 좌표 조정
        h, w = BOX_SIZES[self.current_box][1], BOX_SIZES[self.current_box][0]
        
        cv2.rectangle(state_rgb_with_box, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
        state = state_rgb_with_box
        return state, reward, done, {}

    def _draw(self):
        BOX_SIZES = self.BOX_SIZES
        self.screen.fill((0, 0, 0))
        for index, (box, pos) in enumerate(self.placed_boxes):
            scaled_pos = [coord * scale_factor for coord in pos]
            scaled_size = [size * scale_factor for size in BOX_SIZES[box]]
            pygame.draw.rect(self.screen, self.BOX_COLORS[index], (*scaled_pos, *scaled_size))
        if len(self.placed_boxes) == 5:
            self.no_door_image = pygame.surfarray.array3d(self.screen)
        scaled_current_pos = [coord * scale_factor for coord in self.current_pos]
        scaled_current_size = [size * scale_factor for size in BOX_SIZES[self.current_box]]
        # pygame.draw.rect(self.screen, (0,0,255), (*scaled_current_pos, *scaled_current_size), width = 1)
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return self.step(0)  # move left
                elif event.key == pygame.K_RIGHT:
                    return self.step(1)  # move right
                elif event.key == pygame.K_UP:
                    return self.step(2)  # move up
                elif event.key == pygame.K_DOWN:
                    return self.step(3)  # move down
                elif event.key == pygame.K_SPACE:
                    return self.step(4)  # place
        return None, 0, False, {}
        
# Example game loop
def main():
    game = layoutAI()
    while True:
        state, reward, done, info = game.handle_events()
        if done:
            break
        game.clock.tick(60)

if __name__ == "__main__":
    main()