APPS=readimg 
ARGS=$(shell pkg-config --cflags --libs opencv)

all: $(APPS)

%: %.cpp
	@echo "Compile the cpp files."
	g++ -o $@ $< $(ARGS)

.PHONY: clean
clean:
	@echo "Clean the project."
	-rm -rf $(APPS)


