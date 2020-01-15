all:
	g++ -std=c++0x -O3 -DUSEDEBUG -I./include *.cpp -o  sqj2_tb.exe

trace:
	g++ -std=c++0x -O3 -DTRACE -I./include *.cpp -o  sqj2_tb.exe

debug:
	g++ -std=c++0x -O0 -g -DUSEDEBUG -I./include *.cpp -o  sqj2_tb.exe

clean:
	rm sqj2_tb.exe
