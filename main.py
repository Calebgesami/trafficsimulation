"""
AI-Powered Smart Traffic Management System
------------------------------------------
Hybrid visualization with Pygame road network and Matplotlib analytics
"""

import pygame
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import time
import threading
import joblib
import os
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from collections import deque
import seaborn as sns
import queue
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
INTERSECTIONS = ['A', 'B', 'C', 'D']  # Four intersections in the network
WEATHER_CONDITIONS = ['sunny', 'rainy', 'snowy']
HISTORICAL_FILE = 'traffic_historical.csv'
MODEL_FILE = 'traffic_rf_model.joblib'
SIMULATION_SPEED_FACTOR = 0.1  # Accelerate simulation: 1 real sec = 10 simulated minutes
BASE_GREEN_TIME = 30  # seconds
MAX_GREEN_TIME = 60   # seconds
MIN_GREEN_TIME = 15   # seconds

# Pygame configuration
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
ROAD_WIDTH = 40
INTERSECTION_SIZE = 80
CAR_WIDTH = 20
CAR_HEIGHT = 30
CAR_SPEED = 2
SPAWN_RATE = 0.3  # Probability of spawning a car per frame

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)
DARK_GREEN = (0, 100, 0)
BROWN = (139, 69, 19)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

# Global shared state
current_live_data = {iid: {'timestamp': None, 'vehicle_count': 0, 'weather': 'sunny'} for iid in INTERSECTIONS}
signal_plans = {iid: {'green_duration': BASE_GREEN_TIME, 'reason': 'default', 'current_state': 'green', 'timer': 0} for iid in INTERSECTIONS}
ml_model = None
feature_columns = None
app = Flask(__name__)

# Communication queues
pygame_command_queue = queue.Queue()
pygame_status_queue = queue.Queue()

# =============================================================================
# VEHICLE CLASS FOR PYGAME
# =============================================================================
class Vehicle:
    def __init__(self, start_intersection, end_intersection, road):
        self.start_intersection = start_intersection
        self.end_intersection = end_intersection
        self.road = road  # 'horizontal' or 'vertical'
        self.position = 0.0  # 0 to 1 along the road
        self.speed = CAR_SPEED * random.uniform(0.8, 1.2)
        self.waiting = False
        self.color = random.choice([RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE])

        # Update vehicle count at start intersection
        current_live_data[start_intersection]['vehicle_count'] += 1

    def update(self, traffic_lights):
        """Update vehicle position based on traffic lights"""
        if self.waiting:
            # Check if light is green
            if self.road == 'horizontal':
                if traffic_lights[self.start_intersection]['horizontal'] == 'green':
                    self.waiting = False
            else:
                if traffic_lights[self.start_intersection]['vertical'] == 'green':
                    self.waiting = False
        else:
            # Move forward
            self.position += self.speed / 100.0

            # Check if at intersection and need to stop
            if 0.45 < self.position < 0.55:  # At intersection
                if self.road == 'horizontal':
                    if traffic_lights[self.start_intersection]['horizontal'] != 'green':
                        self.waiting = True
                else:
                    if traffic_lights[self.start_intersection]['vertical'] != 'green':
                        self.waiting = True

        return self.position < 1.0  # Return False if reached destination

# =============================================================================
# PYGAME VISUALIZATION
# =============================================================================
class TrafficPygame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Traffic Management System - Road Network")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 18)

        # Define intersection positions (grid layout)
        self.intersection_pos = {
            'A': (SCREEN_WIDTH//3, SCREEN_HEIGHT//3),
            'B': (2*SCREEN_WIDTH//3, SCREEN_HEIGHT//3),
            'C': (SCREEN_WIDTH//3, 2*SCREEN_HEIGHT//3),
            'D': (2*SCREEN_WIDTH//3, 2*SCREEN_HEIGHT//3)
        }

        # Roads: list of (start, end, type)
        self.roads = [
            ('A', 'B', 'horizontal'),
            ('C', 'D', 'horizontal'),
            ('A', 'C', 'vertical'),
            ('B', 'D', 'vertical')
        ]

        # Vehicles on each road
        self.vehicles = {road: [] for road in ['A_B', 'C_D', 'A_C', 'B_D']}

        # Traffic light states for visualization
        self.traffic_lights = {
            iid: {'horizontal': 'green', 'vertical': 'red', 'timer': BASE_GREEN_TIME}
            for iid in INTERSECTIONS
        }

        self.running = True
        self.current_weather = 'sunny'
        self.simulation_mode = 'ML Optimized'  # 'Baseline' or 'ML Optimized'

    def draw_road(self, start_pos, end_pos, horizontal=True):
        """Draw a road between two intersections"""
        if horizontal:
            # Draw horizontal road
            y = start_pos[1]
            # Road fill
            road_rect = pygame.Rect(start_pos[0], y - ROAD_WIDTH//2,
                                   end_pos[0] - start_pos[0], ROAD_WIDTH)
            pygame.draw.rect(self.screen, GRAY, road_rect)
            # Lane markings
            for x in range(int(start_pos[0]), int(end_pos[0]), 40):
                pygame.draw.rect(self.screen, WHITE, (x, y - 2, 20, 4))
            # Road edges
            pygame.draw.line(self.screen, WHITE,
                           (start_pos[0], y - ROAD_WIDTH//2),
                           (end_pos[0], y - ROAD_WIDTH//2), 2)
            pygame.draw.line(self.screen, WHITE,
                           (start_pos[0], y + ROAD_WIDTH//2),
                           (end_pos[0], y + ROAD_WIDTH//2), 2)
        else:
            # Draw vertical road
            x = start_pos[0]
            # Road fill
            road_rect = pygame.Rect(x - ROAD_WIDTH//2, start_pos[1],
                                   ROAD_WIDTH, end_pos[1] - start_pos[1])
            pygame.draw.rect(self.screen, GRAY, road_rect)
            # Lane markings
            for y in range(int(start_pos[1]), int(end_pos[1]), 40):
                pygame.draw.rect(self.screen, WHITE, (x - 2, y, 4, 20))
            # Road edges
            pygame.draw.line(self.screen, WHITE,
                           (x - ROAD_WIDTH//2, start_pos[1]),
                           (x - ROAD_WIDTH//2, end_pos[1]), 2)
            pygame.draw.line(self.screen, WHITE,
                           (x + ROAD_WIDTH//2, start_pos[1]),
                           (x + ROAD_WIDTH//2, end_pos[1]), 2)

    def draw_intersection(self, iid, pos):
        """Draw an intersection with traffic lights"""
        # Draw intersection square
        rect = pygame.Rect(pos[0] - INTERSECTION_SIZE//2,
                          pos[1] - INTERSECTION_SIZE//2,
                          INTERSECTION_SIZE, INTERSECTION_SIZE)
        pygame.draw.rect(self.screen, DARK_GRAY, rect)
        pygame.draw.rect(self.screen, WHITE, rect, 2)

        # Draw intersection label
        label = self.big_font.render(iid, True, WHITE)
        self.screen.blit(label, (pos[0] - 15, pos[1] - 40))

        # Draw traffic lights
        lights = self.traffic_lights[iid]

        # Horizontal lights (left/right)
        h_color = GREEN if lights['horizontal'] == 'green' else \
                 YELLOW if lights['horizontal'] == 'yellow' else RED
        # Left light
        pygame.draw.circle(self.screen, h_color,
                          (pos[0] - 30, pos[1] - 15), 10)
        pygame.draw.circle(self.screen, BLACK,
                          (pos[0] - 30, pos[1] - 15), 10, 2)
        # Right light
        pygame.draw.circle(self.screen, h_color,
                          (pos[0] + 30, pos[1] + 15), 10)
        pygame.draw.circle(self.screen, BLACK,
                          (pos[0] + 30, pos[1] + 15), 10, 2)

        # Vertical lights (up/down)
        v_color = GREEN if lights['vertical'] == 'green' else \
                 YELLOW if lights['vertical'] == 'yellow' else RED
        # Up light
        pygame.draw.circle(self.screen, v_color,
                          (pos[0] - 15, pos[1] - 30), 10)
        pygame.draw.circle(self.screen, BLACK,
                          (pos[0] - 15, pos[1] - 30), 10, 2)
        # Down light
        pygame.draw.circle(self.screen, v_color,
                          (pos[0] + 15, pos[1] + 30), 10)
        pygame.draw.circle(self.screen, BLACK,
                          (pos[0] + 15, pos[1] + 30), 10, 2)

        # Draw vehicle count
        count = current_live_data[iid]['vehicle_count']
        count_text = self.font.render(f"🚗 {count}", True, WHITE)
        count_bg = pygame.Rect(pos[0] - 30, pos[1] + 20, 60, 25)
        pygame.draw.rect(self.screen, BLACK, count_bg)
        pygame.draw.rect(self.screen, WHITE, count_bg, 1)
        self.screen.blit(count_text, (pos[0] - 25, pos[1] + 25))

    def draw_vehicle(self, vehicle, road_key, start_pos, end_pos):
        """Draw a vehicle on the road"""
        if vehicle.road == 'horizontal':
            # Calculate position along horizontal road
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * vehicle.position
            # Alternate lanes based on position
            y = start_pos[1] - 12 if (int(vehicle.position * 10) % 2 == 0) else start_pos[1] + 12
        else:
            # Calculate position along vertical road
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * vehicle.position
            # Alternate lanes based on position
            x = start_pos[0] - 12 if (int(vehicle.position * 10) % 2 == 0) else start_pos[0] + 12

        # Draw car rectangle
        car_rect = pygame.Rect(x - CAR_WIDTH//2, y - CAR_HEIGHT//2,
                              CAR_WIDTH, CAR_HEIGHT)
        pygame.draw.rect(self.screen, vehicle.color, car_rect)
        pygame.draw.rect(self.screen, BLACK, car_rect, 2)

        # Draw windows
        window_rect = pygame.Rect(x - CAR_WIDTH//2 + 3, y - CAR_HEIGHT//2 + 3,
                                 CAR_WIDTH - 6, CAR_HEIGHT//3)
        pygame.draw.rect(self.screen, LIGHT_BLUE, window_rect)

        # Draw headlights for waiting cars
        if vehicle.waiting:
            pygame.draw.circle(self.screen, YELLOW,
                             (x - 8, y - 8), 3)
            pygame.draw.circle(self.screen, YELLOW,
                             (x + 8, y - 8), 3)

    def draw_weather_effect(self):
        """Draw weather effects"""
        if self.current_weather == 'rainy':
            # Draw rain drops
            for _ in range(50):
                x = (pygame.time.get_ticks() // 10 + random.randint(0, 1000)) % SCREEN_WIDTH
                y = (pygame.time.get_ticks() // 5 + random.randint(0, 1000)) % SCREEN_HEIGHT
                pygame.draw.line(self.screen, LIGHT_BLUE,
                               (x, y), (x, y + 15), 2)
        elif self.current_weather == 'snowy':
            # Draw snow flakes
            for _ in range(30):
                x = (pygame.time.get_ticks() // 20 + random.randint(0, 1000)) % SCREEN_WIDTH
                y = (pygame.time.get_ticks() // 15 + random.randint(0, 1000)) % SCREEN_HEIGHT
                pygame.draw.circle(self.screen, WHITE, (int(x), int(y)), 4)

    def update_traffic_lights(self):
        """Update traffic light states based on signal plans"""
        global signal_plans

        for iid in INTERSECTIONS:
            if iid in signal_plans:
                # Simple cycle: green -> yellow -> red
                if self.traffic_lights[iid]['timer'] <= 0:
                    if self.traffic_lights[iid]['horizontal'] == 'green':
                        self.traffic_lights[iid]['horizontal'] = 'yellow'
                        self.traffic_lights[iid]['timer'] = 5
                    elif self.traffic_lights[iid]['horizontal'] == 'yellow':
                        self.traffic_lights[iid]['horizontal'] = 'red'
                        self.traffic_lights[iid]['vertical'] = 'green'
                        self.traffic_lights[iid]['timer'] = signal_plans[iid]['green_duration']
                    elif self.traffic_lights[iid]['vertical'] == 'green':
                        self.traffic_lights[iid]['vertical'] = 'yellow'
                        self.traffic_lights[iid]['timer'] = 5
                    elif self.traffic_lights[iid]['vertical'] == 'yellow':
                        self.traffic_lights[iid]['vertical'] = 'red'
                        self.traffic_lights[iid]['horizontal'] = 'green'
                        self.traffic_lights[iid]['timer'] = signal_plans[iid]['green_duration']
                else:
                    self.traffic_lights[iid]['timer'] -= 1

    def spawn_vehicle(self):
        """Randomly spawn new vehicles"""
        if random.random() < SPAWN_RATE:
            road_key = random.choice(list(self.vehicles.keys()))

            if road_key == 'A_B':
                start, end = 'A', 'B'
                road_type = 'horizontal'
            elif road_key == 'C_D':
                start, end = 'C', 'D'
                road_type = 'horizontal'
            elif road_key == 'A_C':
                start, end = 'A', 'C'
                road_type = 'vertical'
            else:  # B_D
                start, end = 'B', 'D'
                road_type = 'vertical'

            vehicle = Vehicle(start, end, road_type)
            self.vehicles[road_key].append(vehicle)

    def update_vehicles(self):
        """Update all vehicle positions"""
        for road_key, vehicles in list(self.vehicles.items()):
            remaining_vehicles = []
            for vehicle in vehicles:
                if vehicle.update(self.traffic_lights):
                    remaining_vehicles.append(vehicle)
                else:
                    # Vehicle reached destination, update count
                    dest = vehicle.end_intersection
                    current_live_data[dest]['vehicle_count'] = \
                        max(0, current_live_data[dest]['vehicle_count'] - 1)
            self.vehicles[road_key] = remaining_vehicles

    def draw_status_panel(self):
        """Draw status information panel"""
        # Main status panel
        panel_rect = pygame.Rect(10, 10, 280, 200)
        pygame.draw.rect(self.screen, BLACK, panel_rect)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2)

        y_offset = 20
        title = self.font.render("AI Traffic Control", True, WHITE)
        self.screen.blit(title, (20, y_offset))

        y_offset += 25
        mode_text = f"Mode: {self.simulation_mode}"
        mode_color = GREEN if 'ML' in self.simulation_mode else YELLOW
        mode_line = self.font.render(mode_text, True, mode_color)
        self.screen.blit(mode_line, (20, y_offset))

        y_offset += 25
        weather_text = f"Weather: {self.current_weather.capitalize()}"
        weather_color = LIGHT_BLUE if self.current_weather == 'rainy' else \
                       WHITE if self.current_weather == 'snowy' else YELLOW
        weather_line = self.font.render(weather_text, True, weather_color)
        self.screen.blit(weather_line, (20, y_offset))

        y_offset += 25
        for iid in INTERSECTIONS:
            plan = signal_plans.get(iid, {})
            text = f"{iid}: {plan.get('green_duration', 30)}s green"
            color = GREEN if plan.get('reason') != 'default' else WHITE
            line = self.small_font.render(text, True, color)
            self.screen.blit(line, (20, y_offset))
            y_offset += 18

        # ML Stats panel
        stats_rect = pygame.Rect(10, 220, 280, 120)
        pygame.draw.rect(self.screen, BLACK, stats_rect)
        pygame.draw.rect(self.screen, WHITE, stats_rect, 2)

        y_offset = 240
        stats_title = self.font.render("ML Performance", True, WHITE)
        self.screen.blit(stats_title, (20, y_offset))

        y_offset += 25
        stats = metrics.get_summary_stats()
        if stats:
            imp_text = f"Improvement: {stats.get('improvement', 0):.1f}%"
            imp_color = GREEN if stats.get('improvement', 0) > 0 else RED
            imp_line = self.font.render(imp_text, True, imp_color)
            self.screen.blit(imp_line, (20, y_offset))

            y_offset += 20
            saved_text = f"Time saved: {stats.get('total_saved', 0):.0f}"
            saved_line = self.font.render(saved_text, True, WHITE)
            self.screen.blit(saved_line, (20, y_offset))

        # Controls panel
        controls_rect = pygame.Rect(10, 350, 280, 60)
        pygame.draw.rect(self.screen, BLACK, controls_rect)
        pygame.draw.rect(self.screen, WHITE, controls_rect, 2)

        controls = [
            "S: Change Weather",
            "R: Reset Simulation",
            "M: Toggle Mode"
        ]
        y_offset = 370
        for control in controls:
            ctrl_line = self.small_font.render(control, True, LIGHT_BLUE)
            self.screen.blit(ctrl_line, (20, y_offset))
            y_offset += 18

    def run(self):
        """Main pygame loop"""
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        # Cycle weather
                        weathers = WEATHER_CONDITIONS
                        current_idx = weathers.index(self.current_weather)
                        self.current_weather = weathers[(current_idx + 1) % len(weathers)]
                    elif event.key == pygame.K_r:
                        # Reset simulation
                        self.vehicles = {road: [] for road in self.vehicles.keys()}
                        for iid in INTERSECTIONS:
                            current_live_data[iid]['vehicle_count'] = 0
                    elif event.key == pygame.K_m:
                        # Toggle mode
                        self.simulation_mode = 'Baseline' if self.simulation_mode == 'ML Optimized' else 'ML Optimized'

            # Clear screen
            self.screen.fill(DARK_GREEN)

            # Draw grass texture
            for x in range(0, SCREEN_WIDTH, 50):
                for y in range(0, SCREEN_HEIGHT, 50):
                    color = (random.randint(0, 50), random.randint(80, 150), random.randint(0, 50))
                    pygame.draw.rect(self.screen, color, (x, y, 50, 50))

            # Draw roads
            for start, end, road_type in self.roads:
                self.draw_road(self.intersection_pos[start],
                              self.intersection_pos[end],
                              road_type == 'horizontal')

            # Update traffic lights
            self.update_traffic_lights()

            # Spawn and update vehicles
            self.spawn_vehicle()
            self.update_vehicles()

            # Draw intersections
            for iid, pos in self.intersection_pos.items():
                self.draw_intersection(iid, pos)

            # Draw vehicles
            for road_key, vehicles in self.vehicles.items():
                if road_key == 'A_B':
                    start, end = 'A', 'B'
                elif road_key == 'C_D':
                    start, end = 'C', 'D'
                elif road_key == 'A_C':
                    start, end = 'A', 'C'
                else:  # B_D
                    start, end = 'B', 'D'

                for vehicle in vehicles:
                    self.draw_vehicle(vehicle, road_key,
                                    self.intersection_pos[start],
                                    self.intersection_pos[end])

            # Draw weather effects
            self.draw_weather_effect()

            # Draw status panels
            self.draw_status_panel()

            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS

            # Send status to main thread
            pygame_status_queue.put({
                'weather': self.current_weather,
                'vehicles': {k: len(v) for k, v in self.vehicles.items()},
                'mode': self.simulation_mode
            })

        pygame.quit()

# =============================================================================
# METRICS COLLECTOR (Enhanced)
# =============================================================================
class MetricsCollector:
    def __init__(self, maxlen=100):
        self.timestamps = deque(maxlen=maxlen)
        self.baseline_wait_times = deque(maxlen=maxlen)
        self.ml_wait_times = deque(maxlen=maxlen)
        self.vehicle_counts = {iid: deque(maxlen=maxlen) for iid in INTERSECTIONS}
        self.green_times = {iid: deque(maxlen=maxlen) for iid in INTERSECTIONS}
        self.predictions = {iid: deque(maxlen=maxlen) for iid in INTERSECTIONS}
        self.weather_conditions = deque(maxlen=maxlen)
        self.cumulative_baseline = 0
        self.cumulative_ml = 0
        self.step_count = 0

    def add_step(self, timestamp, baseline_wait, ml_wait, counts, greens, preds, weather):
        self.timestamps.append(timestamp)
        self.baseline_wait_times.append(baseline_wait)
        self.ml_wait_times.append(ml_wait)
        self.weather_conditions.append(weather)

        for iid in INTERSECTIONS:
            self.vehicle_counts[iid].append(counts[iid])
            self.green_times[iid].append(greens[iid])
            self.predictions[iid].append(preds[iid])

        self.cumulative_baseline += baseline_wait
        self.cumulative_ml += ml_wait
        self.step_count += 1

    def get_summary_stats(self):
        if self.step_count == 0:
            return {}

        avg_baseline = self.cumulative_baseline / self.step_count
        avg_ml = self.cumulative_ml / self.step_count
        improvement = ((avg_baseline - avg_ml) / avg_baseline * 100) if avg_baseline > 0 else 0

        return {
            'avg_baseline': avg_baseline,
            'avg_ml': avg_ml,
            'improvement': improvement,
            'total_steps': self.step_count,
            'total_saved': self.cumulative_baseline - self.cumulative_ml
        }

metrics = MetricsCollector(maxlen=200)

# =============================================================================
# DATA GENERATION AND ML MODEL
# =============================================================================
def generate_historical_data(num_rows=10000, output_file=HISTORICAL_FILE):
    """Generate synthetic traffic data with realistic patterns"""
    logging.info(f"Generating {num_rows} rows of historical traffic data...")

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(minutes=15 * i) for i in range(num_rows)]

    data = []
    weekly_pattern = {0: 1.2, 1: 1.1, 2: 1.1, 3: 1.1, 4: 1.3, 5: 0.8, 6: 0.7}

    for dt in dates:
        for intersection_id in INTERSECTIONS:
            timestamp = dt
            day_of_week = dt.weekday()
            hour = dt.hour
            is_holiday = 1 if (day_of_week >= 5 or (dt.month == 12 and dt.day == 25)) else 0
            weather = random.choices(WEATHER_CONDITIONS, weights=[0.7, 0.2, 0.1])[0]

            # Base traffic pattern
            base = 15 * weekly_pattern[day_of_week]

            # Rush hours
            if 7 <= hour <= 9:
                base += 35 * (1 - 0.3 * is_holiday)
            elif 16 <= hour <= 19:
                base += 40 * (1 - 0.2 * is_holiday)
            elif 11 <= hour <= 14:
                base += 15

            # Weather impact
            if weather == 'rainy':
                base *= 0.85
            elif weather == 'snowy':
                base *= 0.6

            # Intersection specific
            if intersection_id == 'A':
                base *= 1.8
            elif intersection_id == 'B':
                base *= 1.2 if hour in range(10, 20) else 0.9
            elif intersection_id == 'C':
                base *= 0.7 if hour in range(9, 17) else 1.1

            vehicle_count = int(max(3, np.random.poisson(base) + np.random.randint(-8, 8)))

            data.append([timestamp, intersection_id, day_of_week, hour,
                        is_holiday, weather, vehicle_count])

    df = pd.DataFrame(data, columns=['timestamp', 'intersection_id', 'day_of_week', 'hour',
                                      'is_holiday', 'weather_condition', 'vehicle_count'])
    df.to_csv(output_file, index=False)
    logging.info(f"Historical data saved to {output_file}")

    print("\n📊 Historical Data Statistics:")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Avg vehicle count: {df['vehicle_count'].mean():.1f}")

    return df

def train_model(data_file=HISTORICAL_FILE):
    """Train a Random Forest model"""
    logging.info("Loading data for model training...")
    df = pd.read_csv(data_file)

    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Create feature columns
    weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather')
    intersection_dummies = pd.get_dummies(df['intersection_id'], prefix='intersection')

    feature_cols = ['day_of_week', 'hour', 'is_holiday', 'month', 'day', 'weekend']
    feature_cols.extend(weather_dummies.columns)
    feature_cols.extend(intersection_dummies.columns)

    X = pd.concat([df[feature_cols[:6]], weather_dummies, intersection_dummies], axis=1)
    y = df['vehicle_count']

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model Performance - MAE: {mae:.2f}, R2: {r2:.3f}")

    # Save model
    joblib.dump((model, feature_cols), MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")

    print(f"\n🤖 Model Training Complete:")
    print(f"   • Mean Absolute Error: {mae:.2f} vehicles")
    print(f"   • R² Score: {r2:.3f}")
    print(f"   • Features used: {len(feature_cols)}")

    return model, feature_cols

def load_model():
    """Load trained model or train if not exists"""
    if os.path.exists(MODEL_FILE):
        logging.info(f"Loading model from {MODEL_FILE}")
        try:
            loaded_data = joblib.load(MODEL_FILE)

            if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                model, features = loaded_data
                logging.info(f"Model loaded successfully with {len(features)} features")
                return model, features
            else:
                logging.warning("Unexpected model format. Training new model...")
                return train_model()
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return train_model()
    else:
        logging.info("Model not found, training new model...")
        return train_model()

def predict_traffic(intersection_id, weather, hour, day_of_week, is_holiday=0):
    """Predict traffic volume using ML model"""
    global ml_model, feature_columns

    if ml_model is None:
        return 25

    try:
        now = datetime.now()

        # Create feature vector
        features = {
            'day_of_week': day_of_week,
            'hour': hour,
            'is_holiday': is_holiday,
            'month': now.month,
            'day': now.day,
            'weekend': 1 if day_of_week >= 5 else 0
        }

        # Add weather features
        for w in WEATHER_CONDITIONS:
            features[f'weather_{w}'] = 1 if weather == w else 0

        # Add intersection features
        for iid in INTERSECTIONS:
            features[f'intersection_{iid}'] = 1 if intersection_id == iid else 0

        # Create dataframe with correct columns
        feature_dict = {}
        for col in feature_columns:
            feature_dict[col] = features.get(col, 0)

        X_pred = pd.DataFrame([feature_dict])
        prediction = ml_model.predict(X_pred)[0]
        return max(0, int(prediction))
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return 25

def optimize_signal_timing(predictions):
    """Optimize traffic signal timing based on predictions"""
    global signal_plans

    new_plans = {}
    total_pred = sum(predictions.values())

    if total_pred == 0:
        for iid in INTERSECTIONS:
            new_plans[iid] = {
                'green_duration': BASE_GREEN_TIME,
                'reason': 'No traffic detected'
            }
    else:
        for iid in INTERSECTIONS:
            pred = predictions.get(iid, 0)
            proportion = pred / total_pred

            green_time = int(BASE_GREEN_TIME * (1 + proportion))
            green_time = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, green_time))

            reason = f"Predicted {pred} vehicles ({proportion*100:.1f}% of total)"
            new_plans[iid] = {
                'green_duration': green_time,
                'reason': reason
            }

    signal_plans = new_plans
    return new_plans

# =============================================================================
# FLASK API ENDPOINTS
# =============================================================================
@app.route('/api/traffic/live', methods=['POST'])
def receive_live_data():
    """Receive real-time traffic data"""
    data = request.get_json()
    intersection_id = data.get('intersection_id')
    if intersection_id and intersection_id in current_live_data:
        current_live_data[intersection_id] = {
            'timestamp': data.get('timestamp'),
            'vehicle_count': data.get('vehicle_count', 0),
            'weather': data.get('weather_condition', 'sunny')
        }
    return jsonify({"status": "success"}), 200

@app.route('/api/traffic/current', methods=['GET'])
def get_current_traffic():
    """Get current traffic conditions"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'intersections': current_live_data
    })

@app.route('/api/traffic/predict', methods=['GET'])
def get_predictions():
    """Get traffic predictions"""
    predictions = {}
    now = datetime.now()
    for iid in INTERSECTIONS:
        weather = current_live_data[iid].get('weather', 'sunny')
        pred = predict_traffic(iid, weather, now.hour, now.weekday())
        predictions[iid] = pred
    return jsonify({'predictions': predictions})

@app.route('/api/control/optimize', methods=['POST'])
def optimize_signals():
    """Optimize traffic signals"""
    data = request.get_json()
    predictions = data.get('predictions', {})
    new_plans = optimize_signal_timing(predictions)
    return jsonify({'signal_plans': new_plans})

@app.route('/api/control/current-plan', methods=['GET'])
def get_current_plan():
    """Get current signal timing plan"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'signal_plans': signal_plans
    })

# =============================================================================
# MATPLOTLIB VISUALIZATION
# =============================================================================
def setup_matplotlib():
    """Set up matplotlib figure"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])  # Wait times
    ax2 = fig.add_subplot(gs[1, 0])  # Vehicle counts
    ax3 = fig.add_subplot(gs[1, 1])  # Metrics table

    return fig, (ax1, ax2, ax3)

def matplotlib_thread():
    """Run matplotlib in a separate thread"""
    fig, axes = setup_matplotlib()

    def update(frame):
        ax1, ax2, ax3 = axes

        # Clear axes
        for ax in axes:
            ax.clear()

        stats = metrics.get_summary_stats()

        # Wait times
        if len(metrics.timestamps) > 0:
            time_labels = [t.strftime('%H:%M:%S') for t in metrics.timestamps]
            min_len = min(len(time_labels), len(metrics.baseline_wait_times), len(metrics.ml_wait_times))

            if min_len > 0:
                ax1.plot(time_labels[-min_len:],
                        list(metrics.baseline_wait_times)[-min_len:],
                        'r-', label='Baseline', linewidth=2)
                ax1.plot(time_labels[-min_len:],
                        list(metrics.ml_wait_times)[-min_len:],
                        'g-', label='ML Optimized', linewidth=2)
                ax1.set_title('Wait Time Comparison')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Wait Time')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)

        # Vehicle counts
        current_counts = [metrics.vehicle_counts[iid][-1] if metrics.vehicle_counts[iid] else 0
                         for iid in INTERSECTIONS]
        colors = ['red' if c > 40 else 'green' for c in current_counts]
        ax2.bar(INTERSECTIONS, current_counts, color=colors)
        ax2.set_title('Current Vehicle Counts')
        ax2.set_ylabel('Vehicles')

        # Metrics table
        ax3.axis('off')
        if stats:
            table_data = [
                ['Metric', 'Value'],
                ['Avg Baseline', f"{stats['avg_baseline']:.2f}"],
                ['Avg ML', f"{stats['avg_ml']:.2f}"],
                ['Improvement', f"{stats['improvement']:.1f}%"],
                ['Time Saved', f"{stats['total_saved']:.2f}"]
            ]
            table = ax3.table(cellText=table_data, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax3.set_title('Performance Metrics')

        plt.tight_layout()
        return axes

    ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show()

# =============================================================================
# SIMULATION ENGINE
# =============================================================================
def simulation_engine():
    """Main simulation engine that coordinates everything"""
    global ml_model, feature_columns, metrics, signal_plans

    with app.test_client() as client:
        # Baseline simulation
        logging.info("Starting baseline simulation...")
        for step in range(30):
            timestamp = datetime.now()

            # Get pygame status
            try:
                status = pygame_status_queue.get_nowait()
                weather = status['weather']
            except queue.Empty:
                weather = 'sunny'

            # Get current counts
            counts = {iid: current_live_data[iid]['vehicle_count'] for iid in INTERSECTIONS}

            # Baseline wait calculation
            total_wait_baseline = sum(counts[iid] * 30 * 0.1 for iid in INTERSECTIONS)

            # Baseline signal plans
            baseline_plans = {iid: {'green_duration': BASE_GREEN_TIME, 'reason': 'baseline'}
                            for iid in INTERSECTIONS}
            signal_plans = baseline_plans

            # Store metrics
            metrics.add_step(
                timestamp=timestamp,
                baseline_wait=total_wait_baseline,
                ml_wait=total_wait_baseline,
                counts=counts,
                greens={iid: BASE_GREEN_TIME for iid in INTERSECTIONS},
                preds={iid: 0 for iid in INTERSECTIONS},
                weather=weather
            )

            time.sleep(0.2)

        # ML-optimized simulation
        logging.info("Starting ML-optimized simulation...")
        for step in range(30):
            timestamp = datetime.now()

            # Get pygame status
            try:
                status = pygame_status_queue.get_nowait()
                weather = status['weather']
            except queue.Empty:
                weather = 'sunny'

            # Get predictions and optimize
            pred_resp = client.get('/api/traffic/predict')
            preds = pred_resp.get_json()['predictions']

            opt_resp = client.post('/api/control/optimize', json={'predictions': preds})
            new_plans = opt_resp.get_json()['signal_plans']

            # Update signal plans
            signal_plans = new_plans

            # Get current counts
            counts = {iid: current_live_data[iid]['vehicle_count'] for iid in INTERSECTIONS}

            # Calculate wait times
            total_wait_ml = 0
            ml_greens = {}
            for iid in INTERSECTIONS:
                green = new_plans[iid]['green_duration']
                ml_greens[iid] = green
                red = 60 - green
                total_wait_ml += counts.get(iid, 0) * red * 0.1

            total_wait_baseline = sum(counts[iid] * 30 * 0.1 for iid in INTERSECTIONS)

            # Store metrics
            metrics.add_step(
                timestamp=timestamp,
                baseline_wait=total_wait_baseline,
                ml_wait=total_wait_ml,
                counts=counts,
                greens=ml_greens,
                preds=preds,
                weather=weather
            )

            time.sleep(0.2)

        # Final stats
        stats = metrics.get_summary_stats()
        logging.info(f"Final Results - Improvement: {stats['improvement']:.1f}%")

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main entry point"""
    global ml_model, feature_columns, metrics

    print("=" * 80)
    print("   AI-POWERED SMART TRAFFIC MANAGEMENT SYSTEM")
    print("   WITH PYGAME ROAD NETWORK AND MATPLOTLIB ANALYTICS")
    print("=" * 80)

    # Generate data if needed
    if not os.path.exists(HISTORICAL_FILE):
        generate_historical_data()

    # Load ML model
    ml_model, feature_columns = load_model()

    # Start Flask in background
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    logging.info("Flask API started on http://localhost:5000")

    # Start matplotlib in background
    matplotlib_thread_instance = threading.Thread(target=matplotlib_thread, daemon=True)
    matplotlib_thread_instance.start()

    # Start simulation engine in background
    sim_thread = threading.Thread(target=simulation_engine, daemon=True)
    sim_thread.start()

    # Start Pygame visualization (main thread)
    print("\n🚦 Starting Pygame road network visualization...")
    print("   Controls:")
    print("   • S - Change weather")
    print("   • R - Reset simulation")
    print("   • M - Toggle mode (Baseline/ML Optimized)")
    print("   • Close Pygame window to stop\n")

    pygame_vis = TrafficPygame()
    pygame_vis.run()

    # After Pygame closes, show final stats
    stats = metrics.get_summary_stats()
    print("\n" + "=" * 80)
    print("FINAL COMPARISON REPORT")
    print("=" * 80)
    print(f"\n📈 Performance Summary:")
    print(f"   • Total simulation steps: {stats['total_steps']}")
    print(f"   • Average wait time (baseline): {stats['avg_baseline']:.2f}")
    print(f"   • Average wait time (ML optimized): {stats['avg_ml']:.2f}")
    print(f"   • Improvement: {stats['improvement']:.1f}%")
    print(f"   • Total time saved: {stats['total_saved']:.2f} units")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
        pygame.quit()