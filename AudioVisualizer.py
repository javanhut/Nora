import pygame
import numpy as np
import sys


class AudioVisualizer:
    def __init__(self, screen_width=800, screen_height=600, fps=60) -> None:
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Real-time Circular Audio Visualizer")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.running = True
        self.audio_data = np.zeros(100)

    def update_audio_data(self, audio_data) -> None:
        self.audio_data = audio_data

    def draw_circular_visualizer(self, center: list, radius: int) -> None:
        num_points = len(self.audio_data)
        angle_step = 2 * np.pi / num_points

        for i in range(num_points):
            mod_radius = (
                radius + self.audio_data[i] * 100
            )  # Modulate radius based on audio data
            x = center[0] + mod_radius * np.cos(i * angle_step)
            y = center[1] + mod_radius * np.sin(i * angle_step)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), 2)

    def run_iteration(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.screen.fill((0, 0, 0))
        self.draw_circular_visualizer(
            (self.screen_width // 2, self.screen_height // 2), 100
        )
        pygame.display.flip()
        self.clock.tick(self.fps)
