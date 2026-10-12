// Low-latency touch injection through the kernel input device.
//
// `input tap` spawns a fresh app_process (Java) per call, which is where the 150-400 ms goes.
// This opens /dev/input/eventN once, stays resident, and writes MT protocol B events directly,
// so a placement costs a couple of writes instead of two process launches.
//
// This is the ordinary Android input path, the same one `input tap` eventually reaches --
// it does NOT touch the game process. No injection, no hooks, no game memory written.
//
// Commands on stdin, one per line:
//   tap X Y                      single tap
//   place X0 Y0 X1 Y1 GAP_MS     card tap, gap, then placement tap (34 ms holds)
//   placeh X0 Y0 X1 Y1 GAP_MS HOLD_MS   same, with the hold per tap given
//   drag X0 Y0 X1 Y1 STEPS STEP_MS      one gesture: down on the card, move, up on the tile
//   taph X Y HOLD_MS             single tap with the hold given
//   d ID X Y / m ID X Y / u ID   one finger going down, moving, coming up, as it happens (a person's touch
//                                relayed from a phone); ID 0-9 is the finger. No reply: nothing waits on these.
//   now                          -> {"now":US}: this device's monotonic clock, the reader's clock
//   at US <command>              the command above, started when the monotonic clock reaches US
//   size W H                     coordinates are screen pixels of a W x H screen, scaled to the device's own range
//                                (an emulator's touchscreen often reports 0..32767); default: written as given
//   pressure                     also send pressure and touch size when the device has those axes (some devices
//                                treat a touch without pressure as hovering); default: not sent
//   version                      -> {"version":5} (2: placeh/drag; 3: taph/now/at, "t0" and "t" in replies;
//                                4: d/m/u, and "auto" for the device; 5: size, pressure)
//   quit
// Prints one line per command with the elapsed microseconds, so latency is measurable; "t0" is the
// monotonic time it started and "t" the time of each touch down and up (the clock of the reader's
// sample_monotonic_us, so a touch can be placed against what the game's memory held when).
//
// usage: fast_tap /dev/input/eventN        (or: fast_tap auto -- the first device with multi-touch positions)
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int fd;
static int screen_w, screen_h;          /* "size": scale screen pixels to the device's range; 0 = as given */
static int max_x, max_y, max_pressure, max_major;
static int send_pressure;               /* "pressure": on when the device has the axes */

static int axis_max(int code) {
  struct input_absinfo info;
  return ioctl(fd, EVIOCGABS(code), &info) >= 0 ? info.maximum : 0;
}

static int sx(int x) { return screen_w > 0 && max_x > 0 ? (int)((long long)x * (max_x + 1) / screen_w) : x; }
static int sy(int y) { return screen_h > 0 && max_y > 0 ? (int)((long long)y * (max_y + 1) / screen_h) : y; }

static uint64_t now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static void emit(uint16_t type, uint16_t code, int32_t value) {
  struct input_event ev;
  memset(&ev, 0, sizeof(ev));
  ev.type = type;
  ev.code = code;
  ev.value = value;
  if (write(fd, &ev, sizeof(ev)) != (ssize_t)sizeof(ev))
    perror("write");
}

static void sync_report(void) { emit(EV_SYN, SYN_REPORT, 0); }

static int tracking = 1;

static void touch_down(int x, int y) {
  emit(EV_ABS, ABS_MT_SLOT, 0);
  emit(EV_ABS, ABS_MT_TRACKING_ID, tracking++);
  emit(EV_ABS, ABS_MT_POSITION_X, sx(x));
  emit(EV_ABS, ABS_MT_POSITION_Y, sy(y));
  if (send_pressure && max_pressure > 0) emit(EV_ABS, ABS_MT_PRESSURE, max_pressure / 2 > 0 ? max_pressure / 2 : 1);
  if (send_pressure && max_major > 0) emit(EV_ABS, ABS_MT_TOUCH_MAJOR, max_major / 8 > 0 ? max_major / 8 : 1);
  emit(EV_KEY, BTN_TOUCH, 1);
  sync_report();
}

static void touch_up(void) {
  emit(EV_ABS, ABS_MT_SLOT, 0);
  emit(EV_ABS, ABS_MT_TRACKING_ID, -1);
  emit(EV_KEY, BTN_TOUCH, 0);
  sync_report();
}

static void touch_move(int x, int y) {
  emit(EV_ABS, ABS_MT_SLOT, 0);
  emit(EV_ABS, ABS_MT_POSITION_X, sx(x));
  emit(EV_ABS, ABS_MT_POSITION_Y, sy(y));
  sync_report();
}

/* Fingers relayed one event at a time: protocol B slots, one per finger. */
static int fingers_down;

static void finger_down(int slot, int x, int y) {
  emit(EV_ABS, ABS_MT_SLOT, slot);
  emit(EV_ABS, ABS_MT_TRACKING_ID, tracking++);
  emit(EV_ABS, ABS_MT_POSITION_X, sx(x));
  emit(EV_ABS, ABS_MT_POSITION_Y, sy(y));
  if (fingers_down++ == 0) emit(EV_KEY, BTN_TOUCH, 1);
  sync_report();
}

static void finger_move(int slot, int x, int y) {
  emit(EV_ABS, ABS_MT_SLOT, slot);
  emit(EV_ABS, ABS_MT_POSITION_X, sx(x));
  emit(EV_ABS, ABS_MT_POSITION_Y, sy(y));
  sync_report();
}

static void finger_up(int slot) {
  emit(EV_ABS, ABS_MT_SLOT, slot);
  emit(EV_ABS, ABS_MT_TRACKING_ID, -1);
  if (fingers_down > 0 && --fingers_down == 0) emit(EV_KEY, BTN_TOUCH, 0);
  sync_report();
}

/* The first input device that reports multi-touch positions (the touchscreen). */
static int find_touchscreen(char *path, size_t size) {
  for (int i = 0; i < 16; ++i) {
    snprintf(path, size, "/dev/input/event%d", i);
    int probe = open(path, O_RDONLY | O_CLOEXEC);
    if (probe < 0) continue;
    unsigned long bits[(ABS_MAX + 8 * sizeof(unsigned long)) / (8 * sizeof(unsigned long))];
    memset(bits, 0, sizeof(bits));
    int found = ioctl(probe, EVIOCGBIT(EV_ABS, sizeof(bits)), bits) >= 0 &&
                (bits[ABS_MT_POSITION_X / (8 * sizeof(unsigned long))] >>
                 (ABS_MT_POSITION_X % (8 * sizeof(unsigned long)))) & 1UL;
    close(probe);
    if (found) return 1;
  }
  return 0;
}

static uint64_t stamps[8];
static int stamp_count;

static void print_stamps(void) {
  printf(",\"t\":[");
  for (int i = 0; i < stamp_count; ++i)
    printf(i ? ",%llu" : "%llu", (unsigned long long)stamps[i]);
  printf("]");
}

/* A touch that goes down and up inside one frame can be dropped, so hold briefly. */
static void tap(int x, int y, int hold_ms) {
  if (stamp_count < 7) stamps[stamp_count++] = now_us();
  touch_down(x, y);
  usleep((useconds_t)hold_ms * 1000);
  if (stamp_count < 8) stamps[stamp_count++] = now_us();
  touch_up();
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/dev/input/event1";
  char found[64];
  if (!strcmp(path, "auto")) {
    if (!find_touchscreen(found, sizeof(found))) { fprintf(stderr, "no multi-touch device\n"); return 2; }
    path = found;
  }
  fd = open(path, O_WRONLY | O_CLOEXEC);
  if (fd < 0) { perror(path); return 2; }
  setvbuf(stdout, NULL, _IONBF, 0);
  max_x = axis_max(ABS_MT_POSITION_X);
  max_y = axis_max(ABS_MT_POSITION_Y);
  max_pressure = axis_max(ABS_MT_PRESSURE);
  max_major = axis_max(ABS_MT_TOUCH_MAJOR);
  printf("{\"ready\":\"%s\",\"max_x\":%d,\"max_y\":%d,\"max_pressure\":%d}\n", path, max_x, max_y, max_pressure);

  char buffer[256];
  while (fgets(buffer, sizeof(buffer), stdin)) {
    char *line = buffer;
    unsigned long long when = 0;
    int skip = 0;
    if (sscanf(line, "at %llu %n", &when, &skip) == 1 && skip > 0) {
      line += skip;
      while (now_us() + 2000 < when) usleep(1000);     /* sleep to within 2 ms, then spin */
      while (now_us() < when) {}
    }
    uint64_t started = now_us();
    stamp_count = 0;
    int x0, y0, x1, y1, gap, hold;
    if (sscanf(line, "placeh %d %d %d %d %d %d", &x0, &y0, &x1, &y1, &gap, &hold) == 6) {
      tap(x0, y0, hold);
      usleep((useconds_t)gap * 1000);
      tap(x1, y1, hold);
      printf("{\"placeh\":[%d,%d,%d,%d],\"gap\":%d,\"hold\":%d,\"us\":%llu,\"t0\":%llu", x0, y0, x1,
             y1, gap, hold, (unsigned long long)(now_us() - started), (unsigned long long)started);
      print_stamps();
      printf("}\n");
    } else if (sscanf(line, "taph %d %d %d", &x0, &y0, &hold) == 3) {
      tap(x0, y0, hold);
      printf("{\"taph\":[%d,%d],\"hold\":%d,\"us\":%llu,\"t0\":%llu", x0, y0, hold,
             (unsigned long long)(now_us() - started), (unsigned long long)started);
      print_stamps();
      printf("}\n");
    } else if (sscanf(line, "d %d %d %d", &gap, &x0, &y0) == 3) {
      finger_down(gap, x0, y0);
    } else if (sscanf(line, "m %d %d %d", &gap, &x0, &y0) == 3) {
      finger_move(gap, x0, y0);
    } else if (sscanf(line, "u %d", &gap) == 1) {
      finger_up(gap);
    } else if (!strncmp(line, "now", 3)) {
      printf("{\"now\":%llu}\n", (unsigned long long)started);
    } else if (sscanf(line, "drag %d %d %d %d %d %d", &x0, &y0, &x1, &y1, &gap, &hold) == 6) {
      /* gap = number of move steps, hold = ms per step */
      int steps = gap < 1 ? 1 : gap;
      touch_down(x0, y0);
      for (int i = 1; i <= steps; ++i) {
        usleep((useconds_t)hold * 1000);
        touch_move(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps);
      }
      usleep((useconds_t)hold * 1000);
      touch_up();
      printf("{\"drag\":[%d,%d,%d,%d],\"steps\":%d,\"step_ms\":%d,\"us\":%llu}\n", x0, y0,
             x1, y1, steps, hold, (unsigned long long)(now_us() - started));
    } else if (sscanf(line, "place %d %d %d %d %d", &x0, &y0, &x1, &y1, &gap) == 5) {
      tap(x0, y0, 34);
      usleep((useconds_t)gap * 1000);
      tap(x1, y1, 34);
      printf("{\"place\":[%d,%d,%d,%d],\"us\":%llu}\n", x0, y0, x1, y1,
             (unsigned long long)(now_us() - started));
    } else if (sscanf(line, "tap %d %d", &x0, &y0) == 2) {
      tap(x0, y0, 34);
      printf("{\"tap\":[%d,%d],\"us\":%llu}\n", x0, y0,
             (unsigned long long)(now_us() - started));
    } else if (sscanf(line, "size %d %d", &x0, &y0) == 2) {
      screen_w = x0;
      screen_h = y0;
      printf("{\"size\":[%d,%d],\"device\":[%d,%d]}\n", screen_w, screen_h, max_x, max_y);
    } else if (!strncmp(line, "pressure", 8)) {
      send_pressure = 1;
      printf("{\"pressure\":%d,\"major\":%d}\n", max_pressure, max_major);
    } else if (!strncmp(line, "version", 7)) {
      printf("{\"version\":5}\n");
    } else if (!strncmp(line, "quit", 4)) {
      break;
    } else {
      printf("{\"error\":\"bad command\"}\n");
    }
  }
  close(fd);
  return 0;
}
