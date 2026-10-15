# C core of seedling.
#   make          -> build/libseedling.so  (optimized; Python loads it with ctypes)
#   make debug    -> C unit tests built with AddressSanitizer + UBSan
#   make test-c   -> build and run the C unit tests
#   make clean

CC     ?= gcc
CFLAGS ?= -O3 -march=native
CFLAGS += -std=gnu17 -fPIC -fopenmp -Wall -Wextra

# No -march=native here, so valgrind can run these binaries too.
DEBUG_FLAGS := -std=gnu17 -O1 -g -fno-omit-frame-pointer -fopenmp -pthread \
               -fsanitize=address,undefined -Wall -Wextra

LIB_SRC := $(wildcard csrc/*.c)
LIB     := build/libseedling.so
C_TESTS := build/test_sum build/test_ring

.PHONY: all debug test-c clean

all: $(LIB)

$(LIB): $(LIB_SRC) csrc/seedling.h | build
	$(CC) $(CFLAGS) -shared -o $@ $(LIB_SRC)

debug: $(C_TESTS)

build/test_%: csrc/tests/test_%.c $(LIB_SRC) csrc/seedling.h | build
	$(CC) $(DEBUG_FLAGS) -Icsrc -o $@ $< $(LIB_SRC)

test-c: debug
	@for t in $(C_TESTS); do echo "== $$t"; ./$$t || exit 1; done

build:
	mkdir -p build

clean:
	rm -rf build
