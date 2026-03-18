CXX=g++ -m64 -march=native
CXXFLAGS=-I../common -Iobjs/ -O3 -Wall -mavx2 -ffp-contract=off
ISPC=ispc
# note: requires AVX2 capable machine
ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64 --pic


APP_NAME=sqrt
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

TASKSYS_CXX=$(COMMONDIR)/tasksys.cpp
TASKSYS_LIB=-lpthread
TASKSYS_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(TASKSYS_CXX:.cpp=.o)))

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		$(MKDIR_OBJ)

clean:
		$(CLEAN_CMD)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/sqrtSerial.o $(OBJDIR)/sqrtSimd.o $(OBJDIR)/sqrt_ispc.o $(PPM_OBJ) $(TASKSYS_OBJ)

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm $(TASKSYS_LIB)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(OBJDIR)/$(APP_NAME)_ispc.h $(COMMONDIR)/CycleTimer.h

$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: %.ispc
		$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h

