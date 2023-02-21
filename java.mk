GRAMMAR=Java
START_RULE=compilationUnit
GEN_DIR=./javaparser

JAVAC=javac
.SUFFIXES:.java .class

.java.class:
	$(JAVAC) $*.java

all: generate

generate: $(wildcard $(GRAMMAR)*.g4) java-parser python3-parser

java-parser: antlr-java java_classes

antlr-java:
	antlr $(wildcard $(GRAMMAR)*.g4) -o $(GEN_DIR)/java

java_classes:
	$(JAVAC) $(wildcard $(GEN_DIR)/java/*.java)

python3-parser: antlr-python3

antlr-python3: 
	antlr -Dlanguage=Python3 -visitor $(wildcard $(GRAMMAR)*.g4) -o $(GEN_DIR)/python3 

.PHONY: all grun generate java-parser python3-parser antlr-java antlr-python3

grun:
	cd $(GEN_DIR)/java/; grun $(GRAMMAR) $(START_RULE) -gui

clean:
	rm -f *.class