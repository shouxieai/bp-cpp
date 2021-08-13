
srcs := $(shell find src -name "*.cpp")
objs := $(srcs:.cpp=.o)
objs := $(objs:src/%=objs/%)
mks  := $(objs:.o=.mk)

current_directory := $(abspath .)
include_paths := $(current_directory)/lean/OpenBLAS0.3.17/include
library_paths := $(current_directory)/lean/OpenBLAS0.3.17/lib
ld_librarys   := openblas pthread

# 把每一个头文件路径前面增加-I，库文件路径前面增加-L，链接选项前面加-l
run_paths     := $(library_paths:%=-Wl,-rpath=%)
include_paths := $(include_paths:%=-I%)
library_paths := $(library_paths:%=-L%)
ld_librarys   := $(ld_librarys:%=-l%)

compile_flags := -std=c++11 -w -g -O3 $(include_paths) -fopenmp
link_flags := $(library_paths) $(ld_librarys) $(run_paths) -lgomp

# 所有的头文件依赖产生的makefile文件，进行include
ifneq ($(MAKECMDGOALS), clean)
-include $(mks)
endif

objs/%.o : src/%.cpp
	@echo 编译$<
	@mkdir -p $(dir $@)
	@g++ -c $< -o $@ $(compile_flags)

workspace/pro : $(objs)
	@echo 链接$@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(link_flags)

objs/%.mk : src/%.cpp
	@mkdir -p $(dir $@)
	@g++ -MM $< -MF $@ -MT $(@:.mk=.o) $(compile_flags)

pro : workspace/pro

run : pro
	@cd workspace && ./pro train

test : pro
	@cd workspace && ./pro test

image : pro
	@cd workspace && ./pro image 5.bmp

train : pro
	@cd workspace && ./pro train

clean :
	@rm -rf workspace/pro objs

.PHONY : pro run debug clean image train test