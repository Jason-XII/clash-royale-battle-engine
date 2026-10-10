// Print the hottest Apple-silicon die temperature (°C) via IOKit HID sensors; no root needed.
// Build: clang -O2 -framework IOKit -framework CoreFoundation -o cputemp cputemp.c
// Usage: ./cputemp       -> max die temperature;  ./cputemp -a -> every sensor
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <string.h>

typedef struct __IOHIDEvent *IOHIDEventRef;
typedef struct __IOHIDServiceClient *IOHIDServiceClientRef;
typedef struct __IOHIDEventSystemClient *IOHIDEventSystemClientRef;
IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
int IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef, CFDictionaryRef);
CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef);
IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
CFStringRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef, CFStringRef);
double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);
enum { kTemperature = 15 };

int main(int argc, char **argv) {
  int all = argc > 1 && strcmp(argv[1], "-a") == 0;
  int page = 0xff00, usage = 5;  // Apple vendor page, temperature sensors
  CFNumberRef keys[2] = {CFNumberCreate(NULL, kCFNumberIntType, &page), CFNumberCreate(NULL, kCFNumberIntType, &usage)};
  CFStringRef names[2] = {CFSTR("PrimaryUsagePage"), CFSTR("PrimaryUsage")};
  CFDictionaryRef match = CFDictionaryCreate(NULL, (const void **)names, (const void **)keys, 2,
                                             &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  IOHIDEventSystemClientRef client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
  IOHIDEventSystemClientSetMatching(client, match);
  CFArrayRef services = IOHIDEventSystemClientCopyServices(client);
  double hottest = 0;
  for (CFIndex i = 0; services && i < CFArrayGetCount(services); ++i) {
    IOHIDServiceClientRef service = (IOHIDServiceClientRef)CFArrayGetValueAtIndex(services, i);
    IOHIDEventRef event = IOHIDServiceClientCopyEvent(service, kTemperature, 0, 0);
    if (!event) continue;
    double celsius = IOHIDEventGetFloatValue(event, kTemperature << 16);
    CFStringRef product = IOHIDServiceClientCopyProperty(service, CFSTR("Product"));
    char name[128] = "?";
    if (product) CFStringGetCString(product, name, sizeof(name), kCFStringEncodingUTF8);
    if (all) printf("%-28s %.1f\n", name, celsius);
    if (strstr(name, "tdie") && celsius > hottest) hottest = celsius;  // PMU tdie* = CPU/GPU die sensors
    if (product) CFRelease(product);
    CFRelease(event);
  }
  if (!all) printf("%.1f\n", hottest);
  return 0;
}
