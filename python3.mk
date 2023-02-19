GRAMMAR=Python3
START_RULE=file_input
GEN_DIR=./python3parser

JAVAC=javac
.SUFFIXES:.java .class

.java.class:
	$(JAVAC) $*.java

all: generate

generate: $(wildcard $(GRAMMAR)*.g4) java-parser python3-parser

java-parser: antlr-java java_base_files java_classes

antlr-java:
	antlr $(wildcard $(GRAMMAR)*.g4) -o $(GEN_DIR)/java

java_classes:
	$(JAVAC) $(wildcard $(GEN_DIR)/java/*.java)

java_base_files:
	# copy the *Base.java files to the GEN_DIR/java directory
	cp $(wildcard $(GRAMMAR)*Base.java) $(GEN_DIR)/java

python3-parser: python3-transform antlr-python3 python3_base_files python3-undo-transform

python3-transform:
	# Run python script "transformGrammar.py" to transform the Java files to Python3
	python transformGrammar.py

antlr-python3: 
	antlr -Dlanguage=Python3 -visitor $(wildcard $(GRAMMAR)*.g4) -o $(GEN_DIR)/python3 

python3_base_files:
	cp $(wildcard $(GRAMMAR)*Base.py) $(GEN_DIR)/python3

python3-undo-transform:
	for file in $(wildcard $(GRAMMAR)*.g4); do mv $${file}.bak $$file; done

.PHONY: all grun generate java-parser python3-parser antlr-java antlr-python3

grun:
	cd $(GEN_DIR)/java/; grun $(GRAMMAR) $(START_RULE) -gui

clean:
	rm -f *.class