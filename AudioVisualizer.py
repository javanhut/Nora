import pygame
import numpy as np
import threading
import queue
from threading import Lock

class AudioVisualizer:
    def __init__(self, screen_width=800, screen_height=600, fps=60, max_queue_size=10):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.DOUBLEBUF)
        self.buffer = pygame.Surface((self.screen_width, self.screen_height))
        pygame.display.set_caption("Real-time Circular Audio Visualizer")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.running = True
        self.audio_data = np.zeros(100)
        self.last_audio_data = np.zeros(100)
        self.window = np.hanning(100)  # Hanning window for smoothing
        self.audio_data_queue = queue.Queue(maxsize=max_queue_size)
        self.data_lock = Lock()  # To ensure thread safety
        self.visualization_thread = threading.Thread(target=self.visualization_loop, daemon=True)
        self.visualization_thread.start()

    def visualization_loop(self):
        while self.running:
            try:
                audio_data = self.audio_data_queue.get(timeout=0.1)
                self.update_audio_data(audio_data)
                self.run_iteration()
            except queue.Empty:
                # Handle empty queue case, for example by logging or silently passing
                pass

    def update_audio_data(self, audio_data):
        with self.data_lock:
            if audio_data.any():
                self.audio_data = np.convolve(audio_data, self.window, mode='same')
            else:
                self.audio_data = self.last_audio_data
            self.last_audio_data = self.audio_data

    def draw_circular_visualizer(self, center, radius):
        num_points = len(self.audio_data)
        angle_step = 2 * np.pi / num_points

        for i in range(num_points):
            mod_radius = radius + self.audio_data[i] * 100  # Scale factor for visualization
            x = center[0] + mod_radius * np.cos(i * angle_step)
            y = center[1] + mod_radius * np.sin(i * angle_step)
            pygame.draw.circle(self.buffer, (255, 255, 255), (int(x), int(y)), 2)

    def run_iteration(self):
        self.buffer.fill((0, 0, 0))
        self.draw_circular_visualizer((self.screen_width // 2, self.screen_height // 2), 100)
        self.screen.blit(self.buffer, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

