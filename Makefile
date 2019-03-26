all:
	python setup.py build_ext 
	cp build/lib.*/*.so ../

clean: 
	rm -rf build *.c
	rm ../*cpython*.so
