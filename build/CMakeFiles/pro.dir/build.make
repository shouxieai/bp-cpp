# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/sxai/bpc++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/sxai/bpc++/build

# Include any dependencies generated for this target.
include CMakeFiles/pro.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pro.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pro.dir/flags.make

CMakeFiles/pro.dir/src/main.cpp.o: CMakeFiles/pro.dir/flags.make
CMakeFiles/pro.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/sxai/bpc++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pro.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pro.dir/src/main.cpp.o -c /data/sxai/bpc++/src/main.cpp

CMakeFiles/pro.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pro.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/sxai/bpc++/src/main.cpp > CMakeFiles/pro.dir/src/main.cpp.i

CMakeFiles/pro.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pro.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/sxai/bpc++/src/main.cpp -o CMakeFiles/pro.dir/src/main.cpp.s

CMakeFiles/pro.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/pro.dir/src/main.cpp.o.requires

CMakeFiles/pro.dir/src/main.cpp.o.provides: CMakeFiles/pro.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/pro.dir/build.make CMakeFiles/pro.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/pro.dir/src/main.cpp.o.provides

CMakeFiles/pro.dir/src/main.cpp.o.provides.build: CMakeFiles/pro.dir/src/main.cpp.o


# Object files for target pro
pro_OBJECTS = \
"CMakeFiles/pro.dir/src/main.cpp.o"

# External object files for target pro
pro_EXTERNAL_OBJECTS =

../workspace/pro: CMakeFiles/pro.dir/src/main.cpp.o
../workspace/pro: CMakeFiles/pro.dir/build.make
../workspace/pro: CMakeFiles/pro.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/sxai/bpc++/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../workspace/pro"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pro.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pro.dir/build: ../workspace/pro

.PHONY : CMakeFiles/pro.dir/build

CMakeFiles/pro.dir/requires: CMakeFiles/pro.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/pro.dir/requires

CMakeFiles/pro.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pro.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pro.dir/clean

CMakeFiles/pro.dir/depend:
	cd /data/sxai/bpc++/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/sxai/bpc++ /data/sxai/bpc++ /data/sxai/bpc++/build /data/sxai/bpc++/build /data/sxai/bpc++/build/CMakeFiles/pro.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pro.dir/depend

