# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\INSA\CLANU\LR_MNIST

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\INSA\CLANU\LR_MNIST-Build

# Include any dependencies generated for this target.
include src/CMakeFiles/mnist_train_lrCgd.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/mnist_train_lrCgd.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/mnist_train_lrCgd.dir/flags.make

src/CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.obj: src/CMakeFiles/mnist_train_lrCgd.dir/flags.make
src/CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.obj: src/CMakeFiles/mnist_train_lrCgd.dir/includes_CXX.rsp
src/CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.obj: D:/INSA/CLANU/LR_MNIST/src/mnist_train_lrCgd.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\INSA\CLANU\LR_MNIST-Build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.obj"
	cd /d D:\INSA\CLANU\LR_MNIST-Build\src && C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\mnist_train_lrCgd.dir\mnist_train_lrCgd.cpp.obj -c D:\INSA\CLANU\LR_MNIST\src\mnist_train_lrCgd.cpp

src/CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.i"
	cd /d D:\INSA\CLANU\LR_MNIST-Build\src && C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\INSA\CLANU\LR_MNIST\src\mnist_train_lrCgd.cpp > CMakeFiles\mnist_train_lrCgd.dir\mnist_train_lrCgd.cpp.i

src/CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.s"
	cd /d D:\INSA\CLANU\LR_MNIST-Build\src && C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\INSA\CLANU\LR_MNIST\src\mnist_train_lrCgd.cpp -o CMakeFiles\mnist_train_lrCgd.dir\mnist_train_lrCgd.cpp.s

# Object files for target mnist_train_lrCgd
mnist_train_lrCgd_OBJECTS = \
"CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.obj"

# External object files for target mnist_train_lrCgd
mnist_train_lrCgd_EXTERNAL_OBJECTS =

src/mnist_train_lrCgd.exe: src/CMakeFiles/mnist_train_lrCgd.dir/mnist_train_lrCgd.cpp.obj
src/mnist_train_lrCgd.exe: src/CMakeFiles/mnist_train_lrCgd.dir/build.make
src/mnist_train_lrCgd.exe: lib/libLRLIB.a
src/mnist_train_lrCgd.exe: src/CMakeFiles/mnist_train_lrCgd.dir/linklibs.rsp
src/mnist_train_lrCgd.exe: src/CMakeFiles/mnist_train_lrCgd.dir/objects1.rsp
src/mnist_train_lrCgd.exe: src/CMakeFiles/mnist_train_lrCgd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\INSA\CLANU\LR_MNIST-Build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mnist_train_lrCgd.exe"
	cd /d D:\INSA\CLANU\LR_MNIST-Build\src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\mnist_train_lrCgd.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/mnist_train_lrCgd.dir/build: src/mnist_train_lrCgd.exe

.PHONY : src/CMakeFiles/mnist_train_lrCgd.dir/build

src/CMakeFiles/mnist_train_lrCgd.dir/clean:
	cd /d D:\INSA\CLANU\LR_MNIST-Build\src && $(CMAKE_COMMAND) -P CMakeFiles\mnist_train_lrCgd.dir\cmake_clean.cmake
.PHONY : src/CMakeFiles/mnist_train_lrCgd.dir/clean

src/CMakeFiles/mnist_train_lrCgd.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\INSA\CLANU\LR_MNIST D:\INSA\CLANU\LR_MNIST\src D:\INSA\CLANU\LR_MNIST-Build D:\INSA\CLANU\LR_MNIST-Build\src D:\INSA\CLANU\LR_MNIST-Build\src\CMakeFiles\mnist_train_lrCgd.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/mnist_train_lrCgd.dir/depend
