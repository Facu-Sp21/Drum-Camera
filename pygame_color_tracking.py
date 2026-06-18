import time
from pathlib import Path

import cv2
import numpy as np
import pygame


CAMERA_INDEX = 1
WINDOW_TITLE = "CameraDrum - Pygame latency test"
HI_HAT_SOUND_FILE = "closedHitHat.wav"
SNARE_SOUND_FILE = "snare.wav"

HIT_LINE_Y = 300
MIN_AREA = 100
MIN_DOWNWARD_SPEED = 900
MAX_DOWNWARD_SPEED = 3200
MIN_HIT_VOLUME = 0.25
HIT_COOLDOWN_SECONDS = 0.15
STICKS = {
    "red": {
        "label": "RED STICK",
        "draw_color": (30, 230, 90),
        "hsv_ranges": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ],
    },
    "blue": {
        "label": "BLUE STICK",
        "draw_color": (255, 170, 30),
        "hsv_ranges": [
            (np.array([95, 100, 70]), np.array([130, 255, 255])),
        ],
    },
}
INSTRUMENTS = {
    "hihat": {"label": "HI-HAT", "sound": HI_HAT_SOUND_FILE},
    "snare": {"label": "SNARE", "sound": SNARE_SOUND_FILE},
}


def clamp(value, minimum, maximum): # Evita que el valor se salga del rango definido por mínimo y máximo.
    return max(minimum, min(value, maximum))


def speed_to_volume(speed): # Asocia la velocidad del movimiento con el volumen del sonido,
    normalized = (speed - MIN_DOWNWARD_SPEED) / (MAX_DOWNWARD_SPEED - MIN_DOWNWARD_SPEED)
    return clamp(MIN_HIT_VOLUME + normalized * (1.0 - MIN_HIT_VOLUME), MIN_HIT_VOLUME, 1.0)


def open_camera(): # Configura y abre la cámara para capturar video.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) # Usa DirectShow en Windows para reducir la latencia
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Intenta reducir la latencia al mínimo posible configurando el tamaño del búfer a 1 frame. 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Configura la resolución de captura a 640x480 píxeles. reduce la cantidad de datos que se procesan, lo que puede ayudar a mejorar la velocidad de procesamiento y reducir la latencia.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60) # Intenta configurar la camara a 60 FPS
    return cap


def find_color_target(hsv, hsv_ranges): # Busca el objetivo mas grande para un color.
    mask = None
    for lower_color, upper_color in hsv_ranges:
        current_mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # define los contornos de las mascaras binarias
    targets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        targets.append((int(x), int(y), int(radius), area))

    if not targets:
        return None

    return max(targets, key=lambda target: target[3])


def find_stick_targets(frame): # Busca la baqueta roja y la azul en el frame capturado.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return {
        stick: find_color_target(hsv, config["hsv_ranges"])
        for stick, config in STICKS.items()
    }


def instrument_from_x(x, width):
    return "hihat" if x < width // 2 else "snare"


def frame_to_surface(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.swapaxes(rgb, 0, 1)
    return pygame.surfarray.make_surface(rgb)


def create_snare_sound():
    sample_rate = pygame.mixer.get_init()[0]
    duration = 0.18
    sample_count = int(sample_rate * duration)
    t = np.linspace(0.0, duration, sample_count, endpoint=False)

    rng = np.random.default_rng(3)
    noise = rng.uniform(-1.0, 1.0, sample_count)
    noise_envelope = np.exp(-t * 22.0)
    body = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t * 28.0)

    wave = (noise * noise_envelope * 0.75) + (body * 0.35)
    wave = np.clip(wave, -1.0, 1.0)
    samples = (wave * 32767).astype(np.int16)
    stereo_samples = np.column_stack((samples, samples))
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_samples))


def load_sound(path, fallback_factory=None):
    if Path(path).exists():
        return pygame.mixer.Sound(path)
    if fallback_factory is not None:
        return fallback_factory()
    raise FileNotFoundError(f"No se encontro el sonido: {path}")


def play_hit(sound, volume):
    channel = sound.play()
    if channel is not None:
        channel.set_volume(volume)


def main():
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=128)
    pygame.init()
    pygame.mixer.set_num_channels(16)

    sounds = {
        "hihat": load_sound(INSTRUMENTS["hihat"]["sound"]),
        "snare": load_sound(INSTRUMENTS["snare"]["sound"], create_snare_sound),
    }

    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")

    success, frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError("La camara no entrego frames.")

    height, width = frame.shape[:2]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)

    previous_positions = {stick: None for stick in STICKS}
    previous_times = {stick: None for stick in STICKS}
    last_hit_times = {
        stick: {instrument: 0.0 for instrument in INSTRUMENTS}
        for stick in STICKS
    }
    speeds = {stick: 0.0 for stick in STICKS}
    hit_volumes = {stick: MIN_HIT_VOLUME for stick in STICKS}
    last_instruments = {stick: None for stick in STICKS}
    targets = {stick: None for stick in STICKS}
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        targets = find_stick_targets(frame)
        now = time.perf_counter()

        for stick, target in targets.items():
            if target is None:
                previous_positions[stick] = None
                previous_times[stick] = None
                speeds[stick] = 0.0
                last_instruments[stick] = None
                continue

            x, y, _radius, _area = target
            instrument = instrument_from_x(x, width)
            last_instruments[stick] = instrument
            previous_y = previous_positions[stick]
            previous_time = previous_times[stick]
            delta_time = 0.0 if previous_time is None else now - previous_time

            if previous_y is not None and delta_time > 0:
                speeds[stick] = (y - previous_y) / delta_time
                crossed_down = previous_y < HIT_LINE_Y <= y

                if (
                    crossed_down
                    and speeds[stick] > MIN_DOWNWARD_SPEED
                    and now - last_hit_times[stick][instrument] > HIT_COOLDOWN_SECONDS
                ):
                    hit_volumes[stick] = speed_to_volume(speeds[stick])
                    play_hit(sounds[instrument], hit_volumes[stick])
                    last_hit_times[stick][instrument] = now

            previous_positions[stick] = y
            previous_times[stick] = now

        surface = frame_to_surface(frame)
        screen.blit(surface, (0, 0))

        pygame.draw.line(screen, (40, 130, 255), (0, HIT_LINE_Y), (width, HIT_LINE_Y), 3)
        pygame.draw.line(screen, (255, 210, 60), (width // 2, HIT_LINE_Y), (width // 2, height), 2)

        detected_count = 0
        for stick, target in targets.items():
            if target is None:
                continue

            detected_count += 1
            x, y, radius, _area = target
            pygame.draw.circle(screen, STICKS[stick]["draw_color"], (x, y), max(radius, 10), 2)

        if detected_count > 0:
            red_instrument = (last_instruments["red"] or "-").upper()
            blue_instrument = (last_instruments["blue"] or "-").upper()
            status = (
                f"Baquetas:{detected_count}/2 "
                f"RED->{red_instrument} {speeds['red']:.0f}px/s Vol:{hit_volumes['red']:.2f} | "
                f"BLUE->{blue_instrument} {speeds['blue']:.0f}px/s Vol:{hit_volumes['blue']:.2f} "
                f"FPS:{clock.get_fps():.1f}"
            )
        else:
            status = f"Sin baquetas roja/azul | FPS:{clock.get_fps():.1f}"

        hihat_label = font.render(INSTRUMENTS["hihat"]["label"], True, (255, 255, 255))
        snare_label = font.render(INSTRUMENTS["snare"]["label"], True, (255, 255, 255))
        red_label = font.render(STICKS["red"]["label"], True, STICKS["red"]["draw_color"])
        blue_label = font.render(STICKS["blue"]["label"], True, STICKS["blue"]["draw_color"])
        screen.blit(hihat_label, (20, HIT_LINE_Y + 12))
        screen.blit(snare_label, (width // 2 + 20, HIT_LINE_Y + 12))
        screen.blit(red_label, (20, HIT_LINE_Y + 42))
        screen.blit(blue_label, (20, HIT_LINE_Y + 72))

        text = font.render(status, True, (255, 255, 255))
        shadow = font.render(status, True, (0, 0, 0))
        screen.blit(shadow, (11, 11))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(120)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
