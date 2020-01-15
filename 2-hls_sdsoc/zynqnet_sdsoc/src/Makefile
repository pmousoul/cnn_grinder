all:
	g++ -std=c++0x -O3 -I./vivado_include *.cpp -o  cpu_top.exe

trace:
	g++ -std=c++0x -O3 -DTRACE -I./vivado_include *.cpp -o  cpu_top.exe

debug:
	g++ -std=c++0x -O0 -g -I./vivado_include *.cpp -o  cpu_top.exe

clean:
	rm cpu_top.exe
