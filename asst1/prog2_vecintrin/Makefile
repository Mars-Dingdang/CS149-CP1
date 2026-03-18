ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
RM_CMD=-del /Q *.o myexp *~ 2>nul
else
RM_CMD=rm -f *.o myexp *~
endif

all: myexp

logger.o: logger.cpp logger.h CS149intrin.h CS149intrin.cpp
	g++ -c logger.cpp

CS149intrin.o: CS149intrin.cpp CS149intrin.h logger.cpp logger.h
	g++ -c CS149intrin.cpp

myexp: CS149intrin.o logger.o main.cpp
	g++ -I../common logger.o CS149intrin.o main.cpp -o myexp

clean:
	$(RM_CMD)
