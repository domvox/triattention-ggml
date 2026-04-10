CC = gcc
CFLAGS = -O2 -Wall -Wextra -I.

all: libtriattention.a bench_tria test_tria_lib

libtriattention.a: triattention.o triattention_score_ref.o
	ar rcs $@ $^

bench_tria: bench_tria.c libtriattention.a
	$(CC) $(CFLAGS) -o $@ $< -L. -ltriattention -lm

test_tria_lib: test_tria_lib.c libtriattention.a
	$(CC) $(CFLAGS) -o $@ $< -L. -ltriattention -lm

%.o: %.c triattention.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o *.a bench_tria test_tria_lib

.PHONY: all clean
