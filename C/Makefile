CC=g++

all: libws_c.so main

main: ws_c.c
	${CC} -o main ws_c.c 

libws_c.so: ws_c.c
	${CC} -o libws_c.so -fPIC -shared ws_c.c -O3

clean: 
	rm -f main libws_c.so

