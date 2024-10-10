import pygame
import numpy as np
import math

pygame.init()
width, height = 1300, 700
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
clock = pygame.time.Clock()

def wave_function_3d(x, y, z, t, sigma=1.0, omega=1.0, k=np.array([1.0, 1.0, 1.0])):
    kx, ky, kz = k
    return np.exp(-((x**2 + y**2 + z**2) / (2 * sigma**2))) * np.exp(-1j * (kx*x + ky*y + kz*z - omega*t))

x, y, z = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
psi_3d = wave_function_3d(x, y, z, 0)
probability_density = np.abs(psi_3d)**2

massive_object_pos = np.array([0, 0, 0])
gravitational_constant = 0.1

particles = np.vstack((x.flatten(), y.flatten(), z.flatten(), probability_density.flatten(), np.zeros_like(x.flatten()), np.zeros_like(y.flatten()), np.zeros_like(z.flatten()))).T

zoom = 50
angle_x, angle_y = 0, 0

def project_3d_to_2d(point, angle_x, angle_y, zoom):
    rotation_y = np.array([[math.cos(math.radians(angle_y)), 0, -math.sin(math.radians(angle_y))],
                           [0, 1, 0],
                           [math.sin(math.radians(angle_y)), 0, math.cos(math.radians(angle_y))]])
    rotation_x = np.array([[1, 0, 0],
                           [0, math.cos(math.radians(angle_x)), -math.sin(math.radians(angle_x))],
                           [0, math.sin(math.radians(angle_x)), math.cos(math.radians(angle_x))]])
    
    rotated_point = np.dot(rotation_y, np.dot(rotation_x, point))
    
    scale = zoom / (5 + rotated_point[2])
    x2d = int(width / 2 + scale * rotated_point[0])
    y2d = int(height / 2 - scale * rotated_point[1])
    return x2d, y2d, scale

def schwarzschild_geodesic(particle, massive_object_pos, gravitational_constant):
    r = np.linalg.norm(particle[:3] - massive_object_pos)
    schwarzschild_radius = 2 * gravitational_constant  # Simplified for visualization
    if r > schwarzschild_radius:
        acceleration = -gravitational_constant / r**2
        unit_vector = (massive_object_pos - particle[:3]) / r
        return acceleration * unit_vector
    else:
        return np.zeros(3)

def run():
    global angle_x, angle_y, zoom
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            angle_y -= 10
        if keys[pygame.K_RIGHT]:
            angle_y += 10
        if keys[pygame.K_UP]:
            angle_x -= 10
        if keys[pygame.K_DOWN]:
            angle_x += 10
        if keys[pygame.K_a]:
            zoom += 25
        if keys[pygame.K_z]:
            zoom -= 25

        screen.fill((0, 0, 0))

        for particle in particles:
            particle[:3] += particle[4:]
            particle[4:] += schwarzschild_geodesic(particle, massive_object_pos, gravitational_constant)

        for particle in particles:
            x2d, y2d, scale = project_3d_to_2d(particle[:3], angle_x, angle_y, zoom)
            size = max(1, int(scale * particle[3] * 10))
            color_intensity = int(particle[3] * 255)
            color = (color_intensity, color_intensity, 255)
            pygame.draw.circle(screen, color, (x2d, y2d), size)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

run()
