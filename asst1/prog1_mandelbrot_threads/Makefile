
CXX=g++ -m64
CXXFLAGS=-I../common -Iobjs/ -O3 -std=c++11 -Wall -fPIC -mavx2

APP_NAME=mandelbrot
OBJDIR=objs
COMMONDIR=../common

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
MKDIR_OBJ=if not exist "$(OBJDIR)" mkdir "$(OBJDIR)"
CLEAN_CMD=-rmdir /S /Q $(OBJDIR) 2>nul & del /Q *.ppm *~ $(APP_NAME) 2>nul
else
MKDIR_OBJ=mkdir -p $(OBJDIR)/
CLEAN_CMD=rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)
endif

PPM_CXX=$(COMMONDIR)/ppm.cpp
PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))


default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		$(MKDIR_OBJ)

clean:
		$(CLEAN_CMD)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/mandelbrotSerial.o $(OBJDIR)/mandelbrotThread.o $(PPM_OBJ)

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm -lpthread

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(COMMONDIR)/CycleTimer.h

